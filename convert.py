#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import subprocess
import logging
from argparse import ArgumentParser
import shutil
from pathlib import Path
from tqdm import tqdm
from glob import glob
import shutil
from PIL import Image
import numpy as np
import multiprocessing as mp

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--has_masks", action='store_true')
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

# configure logging
logging.basicConfig(level = logging.INFO)

# execute a command after logging it and propagate failure correctly
def exec(cmd):
    logging.info(f"Executing: {cmd}")
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, shell=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with code {e.returncode}. Exiting.")
        exit(e.returncode)

def replace_extension(filename, new_extension):
    return os.path.splitext(filename)[0] + new_extension

input_images_path = args.source_path + "/input/images"
distorted_path = args.source_path + "/distorted"

# create output directory
os.makedirs(distorted_path, exist_ok=True)

## Feature extraction
feat_extracton_cmd = colmap_command + " feature_extractor "\
    "--image_path " + input_images_path + " \
    --database_path " + distorted_path + "/database.db \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model " + args.camera + " \
    --SiftExtraction.use_gpu " + str(use_gpu)
exec(feat_extracton_cmd)

## Feature matching
feat_matching_cmd = colmap_command + " exhaustive_matcher \
    --database_path " + distorted_path + "/database.db \
    --SiftMatching.use_gpu " + str(use_gpu)
exec(feat_matching_cmd)

os.makedirs(distorted_path + "/sparse", exist_ok=True)

### Bundle adjustment
# The default Mapper tolerance is unnecessarily large,
# decreasing it speeds up bundle adjustment steps.
mapper_cmd = (colmap_command + " mapper \
    --database_path " + distorted_path + "/database.db \
    --image_path "  + input_images_path + " \
    --output_path "  + distorted_path + "/sparse \
    --Mapper.ba_global_function_tolerance=0.000001")
exec(mapper_cmd)

# select the largest submodel from resulting sparse models
i = 0
largest_size = 0
index = 0

while True:
    path = distorted_path + "/sparse/" + str(i)
    if not os.path.exists(path):
        break

    # check the file size of images.bin
    images_bin = path + "/images.bin"
    size = os.path.getsize(images_bin)
    if size > largest_size:
        largest_size = size
        index = i

    i += 1

str_index = str(index)

sparse_path = distorted_path + "/sparse/" + str_index
oriented_path = distorted_path + "/oriented"
undistorted_path = args.source_path + "/undistorted"

os.makedirs(oriented_path, exist_ok=True)

# orientate the chosen model
aligner_cmd = (colmap_command + " model_orientation_aligner \
    --image_path " + input_images_path + " \
    --input_path " + sparse_path + " \
    --output_path " + oriented_path)
exec(aligner_cmd)

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + input_images_path + " \
    --input_path " + oriented_path + " \
    --output_path " + undistorted_path + "\
    --output_type COLMAP")
exec(img_undist_cmd)

### Handle mask images
if args.has_masks:

    masks_path = args.source_path + "/masks"

    os.makedirs(masks_path, exist_ok=True)

    # convert database to TXT format so we can make images .png
    model_converter_cmd = (colmap_command + " model_converter \
        --input_path " + oriented_path + " \
        --output_path " + masks_path + " \
        --output_type TXT")
    exec(model_converter_cmd)

    # read lines
    with open(masks_path + "/images.txt", 'r') as file:
        lines = file.readlines()

    # replace extensions
    l = 0
    for i in range(len(lines)):
        if lines[i].startswith("#"):
            # skip comments
            continue
        if l % 2 == 0:
            # handle every second line
            words = lines[i].rstrip().split(" ")
            words[-1] = replace_extension(words[-1], ".png")
            lines[i] = " ".join(words) + "\n"
        l += 1

    # write modified images.txt
    with open(masks_path + "/images.txt", 'w') as file:
        file.writelines(lines)

    os.makedirs(masks_path + "/undistorted", exist_ok=True)

    # undistort masks
    mask_undist_cmd = (colmap_command + " image_undistorter \
        --image_path " + args.source_path + "/input/masks \
        --input_path " + masks_path + " \
        --output_path " + masks_path + "/undistorted \
        --output_type COLMAP")
    exec(mask_undist_cmd)

    def combine(color_path, alpha_path, output_path):
        alpha = Image.open(alpha_path).convert('L')
        clr = Image.open(color_path)
        clr.putalpha(alpha)
        clr.save(output_path)

    files = os.listdir(undistorted_path + "/images")
    for file in files:
        mask_file = replace_extension(file, ".png")
        color_image = undistorted_path + "/images/" + file
        mask_image = masks_path + "/undistorted/images/" + mask_file
        output_image = undistorted_path + "/images/" + mask_file
        combine(color_image, mask_image, output_image)
        if mask_file != file:
            os.remove(color_image)

    model_src_path = masks_path + "/undistorted/sparse"
else:
    model_src_path = undistorted_path + "/sparse"

# move all files from sparse into sparse/0, as train.py expects it
files = os.listdir(model_src_path)
os.makedirs(undistorted_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == "0":
        continue
    source_file = os.path.join(model_src_path, file)
    destination_file = os.path.join(undistorted_path, "sparse", "0", file)
    shutil.copy(source_file, destination_file)

# Generate 1/2, 1/4 and 1/8th resized images
if (args.resize):
    print("Copying and resizing...")

    images_path = undistorted_path + "/images"
    images_2_path = images_path + "_2"
    images_4_path = images_path + "_4"
    images_8_path = images_path + "_8"

    os.makedirs(images_2_path, exist_ok=True)
    os.makedirs(images_4_path, exist_ok=True)
    os.makedirs(images_8_path, exist_ok=True)

    # Get the list of files in the source directory
    files = os.listdir(images_path)
    for file in files:
        source_file = os.path.join(images_path, file)
        output_file2 = os.path.join(images_2_path, file)
        output_file4 = os.path.join(images_4_path, file)
        output_file8 = os.path.join(images_8_path, file)

        # generate the resized images in a single call
        generate_thumbnails_cmd = ("convert "
            # resize input file, uses less memory
            f"{source_file}[50%]"
            f" -write mpr:thumb -write {output_file2} +delete"
            f" mpr:thumb -resize 50% -write mpr:thumb -write {output_file4} +delete"
            f" mpr:thumb -resize 50% {output_file8}")
        exec(generate_thumbnails_cmd)

print("Done.")
