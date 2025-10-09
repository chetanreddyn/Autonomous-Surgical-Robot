import os
import cv2
import numpy as np
from pathlib import Path

class VideoCreatorNoCSV:
    def __init__(self, demo_folder, output_folder, fps, resolution):
        """
        Initialize the VideoCreator object to work directly with image folders.

        Args:
            demo_folder (str): Path to the demo folder containing left_images and right_images.
            output_folder (str): Path to the folder where the videos will be saved.
            fps (int): Frames per second for the output videos.
            resolution (tuple): Resolution for the output videos (width, height).
        """
        self.demo_folder = demo_folder
        self.output_folder = output_folder
        self.fps = fps
        self.resolution = resolution
        self.left_video_path = os.path.join(output_folder, "left.mp4")
        self.right_video_path = os.path.join(output_folder, "right.mp4")
        self.left_right_video_path = os.path.join(output_folder, "left_right.mp4")

        # Ensure the output folder exists
        os.makedirs(self.output_folder, exist_ok=True)

    def get_image_paths(self):
        """
        Get sorted lists of image paths from left_images and right_images folders.

        Returns:
            tuple: Two lists containing left and right image paths.
        """
        left_folder = os.path.join(self.demo_folder, "left_images")
        right_folder = os.path.join(self.demo_folder, "right_images")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        left_images = []
        right_images = []
        
        for ext in image_extensions:
            left_images.extend(Path(left_folder).glob(f'*{ext}'))
            left_images.extend(Path(left_folder).glob(f'*{ext.upper()}'))
            right_images.extend(Path(right_folder).glob(f'*{ext}'))
            right_images.extend(Path(right_folder).glob(f'*{ext.upper()}'))
        
        # Sort by numerical order (extract number from filename)
        def extract_number(path):
            # Extract number from filename like "camera_left_123.png"
            import re
            match = re.search(r'(\d+)', path.stem)
            return int(match.group(1)) if match else 0
        
        left_images = sorted(left_images, key=extract_number)
        right_images = sorted(right_images, key=extract_number)
        
        # Convert to strings
        left_image_paths = [str(p) for p in left_images]
        right_image_paths = [str(p) for p in right_images]
        
        if len(left_image_paths) != len(right_image_paths):
            print(f"Warning: Different number of images - Left: {len(left_image_paths)}, Right: {len(right_image_paths)}")
            # Use the minimum length
            min_len = min(len(left_image_paths), len(right_image_paths))
            left_image_paths = left_image_paths[:min_len]
            right_image_paths = right_image_paths[:min_len]
        
        return left_image_paths, right_image_paths

    def create_videos(self):
        """
        Create three videos (left.mp4, right.mp4, left_right.mp4) from the image paths.
        """
        left_image_paths, right_image_paths = self.get_image_paths()
        
        if not left_image_paths:
            print(f"No images found in {self.demo_folder}")
            return False

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
        
        print(f"Videos created for {self.demo_folder}")
        return True