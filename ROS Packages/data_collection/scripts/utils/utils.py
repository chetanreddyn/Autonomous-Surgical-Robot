import os
import cv2
import pandas as pd
import numpy as np

def create_video_from_first_row(base_folder, output_folder, fps, resolution):
    """
    Process each Demo{i} folder, extract the first row of images, and create a video.

    Args:
        base_folder (str): Path to the folder containing Demo{i} subfolders.
        output_folder (str): Path to the folder where the video will be saved.
        fps (int): Frames per second for the output video.
        resolution (tuple): Resolution for the output video (width, height).
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define video writer
    video_path = os.path.join(output_folder, "combined_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 videos
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (resolution[0] * 2, resolution[1]))

    demo_folders = [f for f in os.listdir(base_folder) if f.startswith("Demo")]

    for demo in sorted(demo_folders, key=lambda x: int(x[4:])):  # Sort by Demo number
        demo_path = os.path.join(base_folder, demo)
        csv_path = os.path.join(demo_path, "data.csv")

        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} does not exist. Skipping {demo}.")
            continue

        # Read the first row of the CSV
        try:
            data = pd.read_csv(csv_path)
            first_row = data.iloc[0]
            left_image_path = first_row["camera_left_image_path"]
            right_image_path = first_row["camera_right_image_path"]

            # Read the images
            left_image = cv2.imread(left_image_path)
            right_image = cv2.imread(right_image_path)

            if left_image is None or right_image is None:
                print(f"Warning: Could not read images for {demo}. Skipping...")
                continue

            # Resize images to the specified resolution
            left_image = cv2.resize(left_image, resolution)
            right_image = cv2.resize(right_image, resolution)

            # Combine left and right images side by side
            combined_image = np.hstack((left_image, right_image))

            # Write the combined image to the video
            video_writer.write(combined_image)

        except Exception as e:
            print(f"Error processing {demo}: {e}")

    # Release the video writer
    video_writer.release()
    print(f"Video saved to {video_path}")

if __name__ == "__main__":
    base_folder = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Collaborative Expert Two Handed Object Transfer"
    output_folder = os.path.join(base_folder, "Data Analysis")

    if os.path.exists(base_folder):
        create_video_from_first_row(base_folder, output_folder, fps=5, resolution=(576, 324))
    else:
        print("Enter Correct Path")