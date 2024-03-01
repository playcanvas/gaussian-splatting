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
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--masks_path", type=str)
parser.add_argument("--generate_text_model", action="store_true")
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
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDERR, shell=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with code {e.returncode}. Exiting.")
        exit(e.returncode)

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    exec(feat_extracton_cmd)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exec(feat_matching_cmd)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exec(mapper_cmd)

# select the largest submodel
i = 0
largest_size = 0
index = 0

while True:
    path = args.source_path + "/distorted/sparse/" + str(i)
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
distorted_sparse_path = args.source_path + "/distorted/sparse/" + str_index


### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + distorted_sparse_path + " \
    --output_path " + args.source_path + "\
    --output_type COLMAP")
exec(img_undist_cmd)


# Handle masks

if args.masks_path is not None:
    # We need to modify the colmap database to reference the mask images
    # which are always in png format.
    mask_model_path = args.masks_path + "/model"
    Path(mask_model_path).mkdir(exist_ok=True)

    # First convert model to text format
    model_converter_cmd = (colmap_command + " model_converter \
        --input_path " + distorted_sparse_path + " \
        --output_path " + mask_model_path + " \
        --output_type TXT")
    exec(model_converter_cmd)

    # read images.txt
    with open(mask_model_path + "/images.txt", 'r') as file:
        lines = file.readlines()

    # replace image filenames with png extensions (and keep the list of renames for later)
    filenames = []
    l = 0
    for i in range(len(lines)):
        if lines[i].startswith("#"):
            # skip comments
            continue
        if l % 2 == 0:
            # handle every second line
            words = lines[i].rstrip().split(" ")
            filename = words[-1].split(".")
            filename[-1] = "png"
            new_filename = ".".join(filename)
            filenames.append([words[-1], new_filename])
            words[-1] = new_filename
            lines[i] = " ".join(words) + "\n"
        l += 1

    # write modified images.txt
    with open(mask_model_path + "/images.txt", 'w') as file:
        file.writelines(lines)

    # Undistort mask images
    seg_undist_cmd = (colmap_command + " image_undistorter \
        --image_path " + args.masks_path + " \
        --input_path " + mask_model_path + " \
        --output_path " + args.masks_path + "/undistorted \
        --output_type COLMAP")
    exec(seg_undist_cmd)

    # combine undistorted color and mask images
    def combine(color_path, alpha_path, output_path):
        alpha = Image.open(alpha_path).convert('L')
        clr = Image.open(color_path)
        clr.putalpha(alpha)
        clr.save(output_path)

    for i in range(len(filenames)):
        color_image = args.source_path + "/images/" + filenames[i][0]
        mask_image = args.masks_path + "/undistorted/images/" + filenames[i][1]
        output_image = args.source_path + "/images/" + filenames[i][1]
        combine(color_image, mask_image, output_image)

    # copy the modified database to final location for use in training
    target_path = args.source_path + "/sparse/0"
    Path(target_path).mkdir(exist_ok=True)

    source_path = args.masks_path + "/undistorted/sparse"
    files = os.listdir(source_path)
    for file in files:
        source_file = os.path.join(source_path, file)
        destination_file = os.path.join(target_path, file)
        shutil.move(source_file, destination_file)
else:
    # move all files from sparse into sparse/0, as train.py expects it
    files = os.listdir(args.source_path + "/sparse")
    os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
    # Copy each file from the source directory to the destination directory
    for file in files:
        if file == "0":
            continue
        source_file = os.path.join(args.source_path, "sparse", file)
        destination_file = os.path.join(args.source_path, "sparse", "0", file)
        shutil.move(source_file, destination_file)

if (args.resize):
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)
        output_file2 = os.path.join(args.source_path, "images_2", file)
        output_file4 = os.path.join(args.source_path, "images_4", file)
        output_file8 = os.path.join(args.source_path, "images_8", file)

        # generate the resized images in a single call
        generate_thumbnails_cmd = ("convert "
            # resize input file, uses less memory
            f"{source_file}[50%]"
            f" -write mpr:thumb -write {output_file2} +delete"
            f" mpr:thumb -resize 50% -write mpr:thumb -write {output_file4} +delete"
            f" mpr:thumb -resize 50% {output_file8}")
        exec(generate_thumbnails_cmd)

print("Done.")
