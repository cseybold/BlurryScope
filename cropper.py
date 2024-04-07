import cv2
import numpy as np
import os
import csv
import skimage as ski
from skimage.segmentation import clear_border
import random


spreadsheet = r"C:\Users\DL04\Documents\ganscanslidescoresheet.csv"

def wholeThang(image):

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  '''
  thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 199, 5)
  ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  cleared = clear_border(thresh2)
  im_blur2 = cv2.medianBlur(cleared, 17)
  '''


  bin2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 23, 2)
  im_blur = cv2.GaussianBlur(bin2,(51,51),0)
  ret, bin_img = cv2.threshold(im_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  cleared = clear_border(bin_img)
  im_blur2 = cv2.medianBlur(cleared, 101)


  display_height = 800  # Adjust this value based on your requirements
  display_width = int(image.shape[1] * (display_height / image.shape[0]))
  display_image = cv2.resize(im_blur2, (display_width, display_height))

  cv2.namedWindow('binary_image', cv2.WINDOW_NORMAL)
  cv2.imshow('binary_image', display_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  # flip?
  flIm = None
  while flIm is None:
    flipornot = int(input("Input 0 for original, 1 for hor flip, or 2 for vert flip: "))
    if flipornot == 0:
      flIm = 0
      break
    elif flipornot == 1:
      flIm = 1
      break
    elif flipornot == 2:
      flIm = 2
      break
    else:
      print('Invalid input: please enter 0 or 1')

  if flIm == 1:
      image = cv2.flip(image, 1)
      im_blur2 = cv2.flip(im_blur2, 1)
  elif flIm == 2:
      image = cv2.flip(image, 0)
      im_blur2 = cv2.flip(im_blur2, 0)


  # Find contours in the thresholded image
  contours, _ = cv2.findContours(im_blur2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  image_with_rectangles = image.copy()

  min_contour_area = 200000  # Adjustable threshold
  max_contour_area = image.shape[0]*image.shape[1]/30

  box_count = 0
  coordinates_list = []

  for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > min_contour_area and area < max_contour_area:
      x, y, w, h = cv2.boundingRect(contour)
      cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)

      contour_image = image[y:y + h, x:x + w]

      box_count += 1
      coordinates_list.append((x, y, w, h))

  left_most = min(x for x, y, w, h in coordinates_list)
  right_most = max(x + w for x, y, w, h in coordinates_list)
  top_most = min(y for x, y, w, h in coordinates_list)
  bottom_most = max(y + h for x, y, w, h in coordinates_list)

  grid = image[top_most:bottom_most, left_most:right_most]

  display_image = cv2.resize(grid, (display_width, display_height))
  cv2.namedWindow('no_edge', cv2.WINDOW_NORMAL)
  cv2.imshow('no_edge', display_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  display_image = cv2.resize(image_with_rectangles, (display_width, display_height))
  cv2.namedWindow('cbox', cv2.WINDOW_NORMAL)
  cv2.imshow('cbox', display_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  # Find the closest contour boxes to the bottom-left and bottom-right grid points
  bottom_left_point = (left_most, top_most)   # can be top or bottom
  bottom_right_point = (right_most, top_most)


  #outline crop boxes
  grid_rows = int(input("Enter #rows: "))
  grid_columns = int(input("Enter #columns (maybe +1): "))
  real_cols = int(input("Enter real #columns: "))
  slide_id = input("Enter slide id: ")

  height, width, _ = grid.shape

  # Define the size of each tissue sample
  sample_width = width // grid_columns
  sample_height = height // grid_rows

  rect_im = grid.copy()
  #rect_im_rot = rotated_image.copy()

  # Draw the grid
  for i in range(sample_width, width-sample_width, sample_width):
      cv2.line(rect_im, (i, 0), (i, height), color=(255, 0, 0), thickness=20)
      #cv2.line(rect_im_rot, (i, 0), (i, height), color=(255, 0, 0), thickness=1)
  for i in range(sample_height, height-sample_height, sample_height):
      cv2.line(rect_im, (0, i), (width, i), color=(255, 0, 0), thickness=20)
      #cv2.line(rect_im_rot, (0, i), (width, i), color=(255, 0, 0), thickness=1)


  #sideByside = np.concatenate((rect_im, rect_im_rot), axis=1)
  display_image = cv2.resize(rect_im, (display_width, display_height))
  cv2.namedWindow('cbox', cv2.WINDOW_NORMAL)
  cv2.imshow('cbox', display_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


  #save crops
  #uniqueSlideID = os.path.splitext(filename)[0].split('_r0')[0]

  output_directory = f"grid_images_{uniqueSlideID}"
  sub_dirs = ['0', '1', '2', '3']
  for x in sub_dirs:
    if not os.path.exists(f"{output_directory}/{x}"):
      os.makedirs(f"{output_directory}/{x}")

  output_directory_2 = f"noleak_images_{uniqueSlideID}"
  sub_dirs = ['0', '1', '2', '3']
  for x in sub_dirs:
    if not os.path.exists(f"{output_directory_2}/{x}"):
      os.makedirs(f"{output_directory_2}/{x}")

  cropSize = 512
  final_cat_grid = np.zeros((grid_rows, real_cols, 5, cropSize, cropSize, 3))

  # Open the CSV file
  with open(spreadsheet, "r") as f:
    reader = csv.reader(f)
    for i in range(grid_rows):
      for j in range(real_cols):
        start_y = i * sample_height
        start_x = j * sample_width
        end_y = start_y + sample_height
        end_x = start_x + sample_width

        # Crop the tissue sample from the image
        grid_image = grid[start_y:end_y, start_x:end_x]

        # match ids up to image
        core_id = f"{chr(65 + i)}{j + 1}"
        row_found = False
        f.seek(0)
        for row in reader:
          if row[1] == slide_id and row[2] == core_id:
            score = row[7]
            row_found = True
            break
        if not row_found:
          score = "X"
          continue

        # remove leakage
        crop = grid_image.copy()
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        second_largest_contour = sorted_contours[1]

        lowest = tuple(second_largest_contour[second_largest_contour[:, :, 1].argmax()][0])
        highest = tuple(second_largest_contour[second_largest_contour[:, :, 1].argmin()][0])
        leftmost = tuple(second_largest_contour[second_largest_contour[:, :, 0].argmin()][0])
        rightmost = tuple(second_largest_contour[second_largest_contour[:, :, 0].argmax()][0])
        yl = min(lowest[1], round(crop.shape[0] / 5))
        yh = min((img.shape[1] - highest[1]), round(crop.shape[0] / 5))
        xl = min((img.shape[0] - leftmost[0]), round(crop.shape[1] / 5))
        xr = min(rightmost[0], round(crop.shape[1] / 5))
        good = min(yl, yh, xl, xr)
        if good == yl:
          mask = np.where(np.arange(img.shape[0]) <= yl)
          crop[mask] = 255
        elif good == yh:
          mask = np.where(np.arange(img.shape[0]) >= highest[1])
          crop[mask] = 255
        elif good == xl:
          mask = np.where(np.arange(img.shape[1]) >= leftmost[0])
          crop[:, mask] = 255
        elif good == xr:
          mask = np.where(np.arange(img.shape[1]) <= xr)
          crop[:, mask] = 255

        # crop the crop
        cat5 = []
        yclo = round(grid_image.shape[0] / 4)
        ychi = grid_image.shape[0] - yclo
        for k in range(4):
          ry = random.randint(yclo, ychi - cropSize)
          rx = random.randint(yclo, ychi - cropSize)
          randcrop = grid_image[ry:(ry + cropSize), rx:(rx + cropSize), :]
          cat5.append(randcrop)
        cat5.append(cv2.resize(crop, (cropSize, cropSize), interpolation=cv2.INTER_CUBIC))

        final_cat_grid[i, j] = cat5

        # Check the score and save the image in the corresponding directory
        if score == "0":
          cv2.imwrite(os.path.join(f"{output_directory}/0", f"{uniqueSlideID}_{core_id}_{score}.png"), grid_image)
          cv2.imwrite(os.path.join(f"{output_directory_2}/0", f"{uniqueSlideID}_{core_id}_{score}_noleak.png"), crop)
        elif score == "1":
          cv2.imwrite(os.path.join(f"{output_directory}/1", f"{uniqueSlideID}_{core_id}_{score}.png"), grid_image)
          cv2.imwrite(os.path.join(f"{output_directory_2}/1", f"{uniqueSlideID}_{core_id}_{score}_noleak.png"), crop)
        elif score == "2":
          cv2.imwrite(os.path.join(f"{output_directory}/2", f"{uniqueSlideID}_{core_id}_{score}.png"), grid_image)
          cv2.imwrite(os.path.join(f"{output_directory_2}/2", f"{uniqueSlideID}_{core_id}_{score}_noleak.png"), crop)
        elif score == "3":
          cv2.imwrite(os.path.join(f"{output_directory}/3", f"{uniqueSlideID}_{core_id}_{score}.png"), grid_image)
          cv2.imwrite(os.path.join(f"{output_directory_2}/3", f"{uniqueSlideID}_{core_id}_{score}_noleak.png"), crop)
        else:
          continue

  np.save(uniqueSlideID, final_cat_grid)


image_dir = r"C:\Users\DL04\PycharmProjects\pythonProject\output_images_BR1202a_1_white"
innermost_folder_name = os.path.basename(image_dir)

image_files = sorted(os.listdir(image_dir))
images = [cv2.imread(os.path.join(image_dir, f)) for f in image_files]
images_rotated = [cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) for image in images]

# Resize images to have the same height
min_height = min(image.shape[0] for image in images_rotated)
images_resized = [image[:min_height, :] for image in images_rotated]

# Concatenate images vertically
result = np.concatenate(images_resized, axis=0)
image = result

base_name = os.path.splitext(innermost_folder_name)[0]
inter_name = base_name.split("_white")[0]
uniqueSlideID = inter_name.split("output_images_")[1]
wholeThang(image)

'''
inp_folder = r
the_images = [f for f in os.listdir(inp_folder) if f.lower().endswith('.png')]
for image in the_images:
  slide_nameIn = os.path.splitext(image)[0].split('_white_cat')[0]
  folder_nameIn = os.path.splitext(image)[0]
'''
