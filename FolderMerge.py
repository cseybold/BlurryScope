import os
from PIL import Image
import shutil
from PIL import Image, ImageOps
from PIL.Image import Resampling


def merge_and_resize_folders(source_folder1, source_folder2, destination_folder, size=(256, 256)):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    subfolders = set(os.listdir(source_folder1)) | set(os.listdir(source_folder2))

    for subfolder in subfolders:
        source_path1 = os.path.join(source_folder1, subfolder)
        source_path2 = os.path.join(source_folder2, subfolder)
        destination_path = os.path.join(destination_folder, subfolder)

        # Create the subfolder in the destination folder if it doesn't exist
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        # Function to copy and resize images
        def copy_and_resize(source_path, destination_path):
            for filename in os.listdir(source_path):
                source_file = os.path.join(source_path, filename)
                destination_file = os.path.join(destination_path, filename)

                # Open, resize, and save the image, overwriting existing files
                with Image.open(source_file) as img:
                    img = img.resize(size, Resampling.LANCZOS)
                    img.save(destination_file)

        # Copy and resize images from both source folders
        if os.path.exists(source_path1):
            copy_and_resize(source_path1, destination_path)

        if os.path.exists(source_path2):
            copy_and_resize(source_path2, destination_path)


# Example usage
source_folder1 = r"C:\Users\DL04\PycharmProjects\pythonProject\grid_images"
source_folder2 = r"C:\Users\DL04\PycharmProjects\downsamplecrop\grid_images"
destination_folder = r"C:\tissuecrops"

merge_and_resize_folders(source_folder1, source_folder2, destination_folder)
