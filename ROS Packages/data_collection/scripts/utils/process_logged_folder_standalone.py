#!/usr/bin/env python

import os
import sys
import argparse
import logging

from frames_to_video import VideoCreator
from frames_to_video_no_csv import VideoCreatorNoCSV
from plotter import plot_arm_joints_and_xyz

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_demo_folder(folder_path):
    """Process a single demo folder"""
    logger.info(f"Processing demo folder: {folder_path}")
    
    csv_path = os.path.join(folder_path, "data.csv")
    actions_csv = os.path.join(folder_path, "teleop_log.csv")
    videos_folder = os.path.join(folder_path, "videos")
    
    try:
        # Check if data.csv exists
        if os.path.exists(csv_path):
            # Generate videos from CSV
            logger.info(f"Generating videos in {videos_folder} from {csv_path} ...")
            video_creator = VideoCreator(csv_path, videos_folder, fps=30, resolution=(576, 324))
            video_creator.create_videos()
            logger.info("Videos generated from CSV.")
            
            # Generate plot.html if we have the CSV
            logger.info(f"Generating plot.html in {folder_path} ...")
            plot_arm_joints_and_xyz(
                data_csv=csv_path,
                actions_csv=actions_csv,
                x_axis="Frame Number",
                title=folder_path,
                logging_folder=folder_path
            )
            logger.info("plot.html generated.")
        else:
            # No CSV, try to create videos directly from images
            logger.info(f"No data.csv found, creating videos from images directly...")
            
            # Check if left_images and right_images directories exist
            left_images_path = os.path.join(folder_path, "left_images")
            right_images_path = os.path.join(folder_path, "right_images")
            
            has_left = os.path.exists(left_images_path) and os.path.isdir(left_images_path)
            has_right = os.path.exists(right_images_path) and os.path.isdir(right_images_path)
            
            if not has_left and not has_right:
                logger.warning(f"No left_images or right_images directories found in {folder_path}, skipping...")
                return False
            
            # Remove any existing videos folder to replace old videos
            if os.path.exists(videos_folder):
                import shutil
                shutil.rmtree(videos_folder)
                logger.info(f"Removed existing videos folder: {videos_folder}")
            
            # Use VideoCreatorNoCSV to create videos in the same format as the original
            logger.info(f"Generating videos in {videos_folder} from image folders...")
            video_creator = VideoCreatorNoCSV(folder_path, videos_folder, fps=30, resolution=(576, 324))
            success = video_creator.create_videos()
            
            if success:
                logger.info("Videos generated successfully.")
            else:
                logger.warning("Failed to generate videos.")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {folder_path}: {str(e)}")
        return False

def find_demo_folders(root_folder):
    """Recursively find all folders with 'demo' or 'Demo' prefix"""
    demo_folders = []
    
    for root, dirs, files in os.walk(root_folder):
        for dir_name in dirs:
            if dir_name.startswith("demo") or dir_name.startswith("Demo"):
                demo_folders.append(os.path.join(root, dir_name))
    
    return demo_folders

def main():
    parser = argparse.ArgumentParser(description='Process demo folders and generate videos/plots')
    parser.add_argument('--folder', '-f', type=str, required=True, 
                        help='Root folder to search for demo folders')
    parser.add_argument('--specific', '-s', type=str, default=None,
                        help='Process a specific folder instead of searching for demo folders')
    
    args = parser.parse_args()
    
    logging_folder = args.folder
    logging_description = args.specific

    if not os.path.exists(logging_folder):
        logger.error(f"Folder does not exist: {logging_folder}")
        return 1

    # If logging_description is provided, process that specific folder
    if logging_description is not None:
        folder_path = os.path.join(logging_folder, logging_description)
        if os.path.exists(folder_path):
            success = process_demo_folder(folder_path)
            return 0 if success else 1
        else:
            logger.error(f"Folder not found: {folder_path}")
            return 1
    else:
        # Otherwise, find and process all Demo* folders recursively
        logger.info(f"Searching for demo* folders in {logging_folder}...")
        demo_folders = find_demo_folders(logging_folder)
        
        if not demo_folders:
            logger.warning(f"No demo* folders found in {logging_folder}")
            return 1
        
        logger.info(f"Found {len(demo_folders)} demo* folders to process")
        
        success_count = 0
        for i, folder in enumerate(demo_folders, 1):
            logger.info(f"\n[{i}/{len(demo_folders)}] Processing: {folder}")
            if process_demo_folder(folder):
                success_count += 1
        
        logger.info(f"\nProcessing complete! Successfully processed {success_count}/{len(demo_folders)} folders.")
        return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())