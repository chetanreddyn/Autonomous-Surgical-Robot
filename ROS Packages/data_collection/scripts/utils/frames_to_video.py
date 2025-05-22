import os
import cv2
import pandas as pd
import numpy as np
import time

class VideoCreator:
    def __init__(self, csv_path, output_folder, fps, resolution):
        """
        Initialize the VideoCreator object.

        Args:
            csv_path (str): Path to the CSV file containing image paths.
            output_folder (str): Path to the folder where the videos will be saved.
            fps (int): Frames per second for the output videos.
            resolution (tuple): Resolution for the output videos (width, height).
        """
        self.csv_path = csv_path
        self.output_folder = output_folder
        self.fps = fps
        self.resolution = resolution
        self.left_video_path = os.path.join(output_folder, "left.mp4")
        self.right_video_path = os.path.join(output_folder, "right.mp4")
        self.left_right_video_path = os.path.join(output_folder, "left_right.mp4")

        # Ensure the output folder exists
        os.makedirs(self.output_folder, exist_ok=True)

    def read_csv(self):
        """
        Read the CSV file and extract left and right image paths.

        Returns:
            tuple: Two lists containing left and right image paths.
        """
        data = pd.read_csv(self.csv_path)
        left_image_paths = data["camera_left_image_path"].tolist()
        right_image_paths = data["camera_right_image_path"].tolist()

        if len(left_image_paths) != len(right_image_paths):
            raise ValueError("The number of left and right images must be the same.")

        return left_image_paths, right_image_paths

    def create_videos(self):
        """
        Create three videos (left.mp4, right.mp4, left_right.mp4) from the image paths.
        """
        left_image_paths, right_image_paths = self.read_csv()

        # Define video writers
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 videos
        left_writer = cv2.VideoWriter(self.left_video_path, fourcc, self.fps, self.resolution)
        right_writer = cv2.VideoWriter(self.right_video_path, fourcc, self.fps, self.resolution)
        left_right_writer = cv2.VideoWriter(self.left_right_video_path, fourcc, self.fps, (self.resolution[0] * 2, self.resolution[1]))

        for left_image_path, right_image_path in zip(left_image_paths, right_image_paths):
            # Read left and right images
            left_image = cv2.imread(left_image_path)
            right_image = cv2.imread(right_image_path)

            if left_image is None or right_image is None:
                print(f"Warning: Could not read images {left_image_path} or {right_image_path}. Skipping...")
                continue

            # Resize images to the specified resolution
            left_image = cv2.resize(left_image, self.resolution)
            right_image = cv2.resize(right_image, self.resolution)

            # Write to left.mp4 and right.mp4
            left_writer.write(left_image)
            right_writer.write(right_image)

            # Combine left and right images side by side for left_right.mp4
            combined_image = np.hstack((left_image, right_image))
            left_right_writer.write(combined_image)

        # Release video writers
        left_writer.release()
        right_writer.release()
        left_right_writer.release()

        # print(f"Videos saved to {self.output_folder}:")
        # print(f"  - Left video: {self.left_video_path}")
        # print(f"  - Right video: {self.right_video_path}")
        # print(f"  - Left-Right video: {self.left_right_video_path}")

if __name__ == "__main__":
    # Specify the path to the CSV file and the output folder
    LOGGING_FOLDER = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Rollouts Autonomous"

    if not os.path.exists(LOGGING_FOLDER):
        print("Enter Correct Logging Folder")
    else:
        t0 = time.time()
        t_prev = t0
        for i in range(1,11):
            logging_description = "Test" + str(i)
            csv_path = os.path.join(LOGGING_FOLDER, logging_description, "data.csv")
            output_folder = os.path.join(LOGGING_FOLDER, logging_description, "videos")

            # Create a VideoCreator object and generate the videos
            video_creator = VideoCreator(csv_path, output_folder, fps=30, resolution=(576, 324))
            video_creator.create_videos()

            t = time.time()
            time_stamp = t-t0
            print(f"Time: {time_stamp:4.3f} | Time Taken: {t-t_prev:4.3} | {logging_description} done")
            t_prev = t
