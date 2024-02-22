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
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)


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

exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"image_undistorter failed with code {exit_code}. Exiting.")
    exit(exit_code)


def remove_dir_if_exist(path):
    if Path(path).exists():
        shutil.rmtree(path)


if args.masks_path is not None:
    remove_dir_if_exist(args.source_path + "/alpha_distorted_sparse_txt/")
    Path(args.source_path + "/alpha_distorted_sparse_txt/").mkdir(exist_ok=True)
    # We need to "hack" colmap to undistort segmentation maps modify paths
    # First convert model to text format
    model_converter_cmd = (colmap_command + " model_converter \
        --input_path " + distorted_sparse_path + " \
        --output_path " + args.source_path + "/alpha_distorted_sparse_txt/ \
        --output_type TXT")
    exit_code = os.system(model_converter_cmd)
    if exit_code != 0:
        logging.error(f"model_converter failed with code {exit_code}. Exiting.")
        exit(exit_code)

    # replace '.jpg' to '.png'
    with open(args.source_path + "/alpha_distorted_sparse_txt/images.txt", "r+") as f:
        images_txt = f.read()
        images_txt = images_txt.replace('.jpg', '.png')
        f.seek(0)
        f.write(images_txt)
        f.truncate()

    # Undistort alpha masks
    seg_undist_cmd = (colmap_command + " image_undistorter \
        --image_path " + args.masks_path + " \
        --input_path " + args.source_path + "/alpha_distorted_sparse_txt/ \
        --output_path " + args.source_path + "/alpha_undistorted_sparse \
        --output_type COLMAP")
    exit_code = os.system(seg_undist_cmd)
    if exit_code != 0:
        logging.error(f"image_undistorter for segs failed with code {exit_code}. Exiting.")
        exit(exit_code)

    # switch images
    remove_dir_if_exist(f'{args.source_path}/alpha_undistorted_sparse/alphas')
    Path(f'{args.source_path}/alpha_undistorted_sparse/images').replace(f'{args.source_path}/alpha_undistorted_sparse/alphas')
    remove_dir_if_exist(f'{args.source_path}/images_src/')
    Path(f'{args.source_path}/images/').replace(f'{args.source_path}/images_src/')

    # concat undistorted images with undistorted alpha masks - TODO: make parallel
    remove_dir_if_exist(f'{args.source_path}/images/')
    Path(f'{args.source_path}/images/').mkdir()

    def concat_alpha(seg_path):
        seg = Image.open(seg_path).convert('L')
        img = Image.open(f'{args.source_path}/images_src/{Path(seg_path).stem}.jpg')
        img.putalpha(seg)
        img.save(f'{args.source_path}/images/{Path(seg_path).stem}.png')

    all_masks_paths = glob(args.source_path + "/alpha_undistorted_sparse/alphas/*.png")
    with mp.Pool() as pool:
        list(tqdm(pool.imap_unordered(concat_alpha, all_masks_paths), total=len(all_masks_paths)))

    # switch models
    remove_dir_if_exist(f'{args.source_path}/sparse_src/')
    Path(f'{args.source_path}/sparse').replace(f'{args.source_path}/sparse_src/')
    Path(f'{args.source_path}/alpha_undistorted_sparse/sparse').replace(f'{args.source_path}/sparse/')


### Convert model to text format so we can read cameras
convert_cmd = (colmap_command + " model_converter \
    --input_path " + args.source_path + "/sparse" + " \
    --output_path "  + args.source_path + "/sparse" + " \
    --output_type TXT")
exit_code = os.system(convert_cmd)
if exit_code != 0:
    logging.error(f"Convert failed with code {exit_code}. Exiting.")
    exit(exit_code)

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

if(args.resize):
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

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system("mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system("mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system("mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")
