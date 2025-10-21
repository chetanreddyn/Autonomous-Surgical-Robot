
import cv2
import os
from glob import glob
import re
from datetime import datetime

def get_image_filenames(images_dir, camera_name):
    """
    Returns a list of image files for a camera, sorted by timestamp in filename.

    Args:
        images_dir (str): Path to the images directory.
        camera_name (str): Camera prefix (e.g., 'camera_left').

    Returns:
        list: Sorted list of image file paths.
    """
    pattern = os.path.join(images_dir, f"{camera_name}_*.png")
    filenames = glob(pattern)
    def extract_timestamp(f):
        match = re.search(rf"{camera_name}_(\d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}}\.\d+)", os.path.basename(f))
        if match:
            return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")
        else:
            raise Exception("regex pattern for images inside the images folder did not match | check extract_timestamp(f) function")
    filenames.sort(key=extract_timestamp)

    return filenames

def create_video_from_images(image_files, output_path, fps=30):
    """
    Creates and saves an MP4 video from a sorted list of image files.

    Args:
        image_files (list): List of image file paths.
        output_path (str): Path to save the output video.
        fps (int): Frames per second for the video.
    """
    if not image_files:
        print(f"No images found for {output_path}")
        return

    # Read first image to get frame size
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MPEG-4 codec
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_file in image_files:
        frame = cv2.imread(img_file)
        video.write(frame)

    video.release()
    print(f"Video saved to {output_path}")

def generate_videos(demo_dir, cameras=['camera_left', 'camera_right'], fps=30):
    """
    Generates and saves MP4 videos for each camera in a demo directory.

    Args:
        demo_dir (str): Path to the demo directory.
        cameras (list): List of camera names.
        fps (int): Frames per second for the video.
    """
    images_dir = os.path.join(demo_dir, "images")
    output_videos_dir = os.path.join(demo_dir, "videos")
    os.makedirs(output_videos_dir, exist_ok=True)
    for camera in cameras:
        image_filenames = get_image_filenames(images_dir, camera)
        output_path = os.path.join(output_videos_dir, f"{camera}.mp4")
        create_video_from_images(image_filenames, output_path, fps)

def generate_videos_for_all_demos(exp_dir, demo_start, demo_end, cameras=['camera_left', 'camera_right'], fps=30):
    """
    Generates videos for all demos in the experiment directory at the specified demo indices

    Args:
        exp_dir (str): Path to the experiment directory.
        demo_start (int): Starting demo index.
        demo_end (int): Ending demo index.
        cameras (list): List of camera names.
        fps (int): Frames per second for the video.
    """
    for i in range(demo_start, demo_end + 1):
        demo_dir = os.path.join(exp_dir, f"Demo{i}")
        if not os.path.exists(demo_dir):
            print(f"Demo directory not found: {demo_dir}")
            continue
        print(f"Generating videos for Demo{i}")
        generate_videos(demo_dir, cameras=cameras, fps=fps)

if __name__ == "__main__":   
    exp_dir = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Two Handed Needle Transfer"
    demo_start = 1
    demo_end = 5
    generate_videos_for_all_demos(exp_dir, demo_start=demo_start, demo_end=demo_end)