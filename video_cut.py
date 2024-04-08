import cv2
import os

def cut_vid(video_path, fps):

  cap = cv2.VideoCapture(video_path)

  # Check if video opened successfully
  if not cap.isOpened():
      print("Error opening video!")
      return

  # Get frame width and height (assuming 480p resolution and 4:3 aspect ratio)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  if width != 640 or height != 480:
      print("Warning: Video resolution is not confirmed to be 480p (640x480).")

  # Define the number of frames for 10 minutes
  total_frames = int(fps * 60 * 10)

  # Define output video writer
  out = cv2.VideoWriter(f"{base_name}_cut.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

  # Process video frames
  count = 0
  while count < total_frames:
      ret, frame = cap.read()

      if not ret:
          print("Error reading frame!")
          break

      # Write the frame to the output video
      out.write(frame)
      count += 1

  # Release resources
  cap.release()
  out.release()
  cv2.destroyAllWindows()


video_path = r"C:\Users\DL04\Documents\tempRedoStitch\BR20826_2_white.mp4"
innermost_folder_name = os.path.basename(video_path)
base_name = os.path.splitext(innermost_folder_name)[0]

fps = 30

cut_vid(video_path, fps)
