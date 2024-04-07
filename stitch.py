#Algorithm is linked from MATLAB

import csv
import os
import shutil
import cv2
import skimage as ski
from skimage.segmentation import clear_border
import numpy as np
import math

import ImageStitch
# Import the matlab module only after you have imported
# MATLAB Compiler SDK generated Python modules.
#import matlab


inp_folder = r"C:\Users\DL04\PycharmProjects\vidwhitebal"
video_files = [f for f in os.listdir(inp_folder) if f.lower().endswith('.mp4')]
for video_file in video_files:
    # Construct the path to the video file
    video_path = os.path.join(inp_folder, video_file)
    print(f"Processing video: {video_path}")


    spreadsheet = r"C:\Users\DL04\Documents\ganscanslidescoresheet.csv"
    videoIn = video_path
    slide_nameIn = os.path.splitext(video_file)[0].split('_')[0]
    print(f"Slide name: {slide_nameIn}")
    folder_nameIn = os.path.splitext(video_file)[0]

    print(f"Extracted slide name: {slide_nameIn}")
    print(f"Extracted folder/slide name: {folder_nameIn}")


    output_folder = os.path.join(os.getcwd(), f"output_images_{folder_nameIn}")
    os.makedirs(output_folder, exist_ok=True)

    my_ImageStitch = ImageStitch.initialize()

    my_ImageStitch.stitchfcnpytry(videoIn, slide_nameIn, nargout=0)

    # Move the images to the created folder
    for i in range(4):
        image_filename = f"{slide_nameIn}_{i}_r0.tif"
        shutil.move(image_filename, os.path.join(output_folder, image_filename))

    my_ImageStitch.terminate()

    # break

    image_dir = output_folder

    image_files = sorted(os.listdir(image_dir))

    images = [cv2.imread(os.path.join(image_dir, f)) for f in image_files]

    images_rotated = [cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) for image in images]

    # Resize images to have the same height
    min_height = min(image.shape[0] for image in images_rotated)
    images_resized = [image[:min_height, :] for image in images_rotated]

    # Concatenate images vertically
    result = np.concatenate(images_resized, axis=0)
    image = result


    display_height = 10000  # Adjust this value based on your requirements
    display_width = int(image.shape[1] * (display_height / image.shape[0]))
    display_image = cv2.resize(result, (display_width, display_height))

    out_temp = os.path.join(os.getcwd(), f"catIm_{folder_nameIn}")
    os.makedirs(out_temp, exist_ok=True)
    cv2.imwrite(os.path.join(f"{out_temp}", f"{folder_nameIn}_cat.png"), display_image)
