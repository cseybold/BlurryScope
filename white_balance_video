import cv2
import numpy as np
import os


inp_folder = r"I:\BR_Scan_1_18_24"
video_files = [os.path.join(inp_folder, f) for f in os.listdir(inp_folder) if f.lower().endswith('.mp4')]

for video_file in video_files:
    # Extract the video name (without extension) from the path
    video_name = os.path.splitext(os.path.basename(video_file))[0]

    # Load the video file
    cap1 = cv2.VideoCapture(video_file)

    frame_number = 100
    cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap1.read()

    if ret:
        # Display the frame
        cv2.imshow("Specific Frame", frame)
        cv2.waitKey(0)  # Wait for a key press
    else:
        print(f"Error reading frame {frame_number} from {video_name}.mp4")

    # Release the video capture
    cap1.release()
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture(video_file)

    imup = frame
    mask = cv2.imread(imup)

    # Get video properties (frame width, height, etc.)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f"{video_name}_white.mp4", fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        diff1 = abs(frame - mask - 1)
        diff2 = abs(mask - frame - 1)
        corrected_frame = np.maximum(diff1, diff2)

        # Write the corrected frame to the output video
        out.write(corrected_frame)

    # Release video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("White balancing applied to all frames. Output saved as 'output_video.mp4'")
