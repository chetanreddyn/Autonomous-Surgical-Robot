#!/usr/bin/env python

import os
import rospy
import glob

from frames_to_video import VideoCreator
from plotter import plot_arm_joints_and_xyz

def process_demo_folder(folder_path):
    """Process a single demo folder"""
    rospy.loginfo(f"Processing demo folder: {folder_path}")
    
    csv_path = os.path.join(folder_path, "data.csv")
    actions_csv = os.path.join(folder_path, "teleop_log.csv")
    videos_folder = os.path.join(folder_path, "videos")
    
    # Check if data.csv exists
    if not os.path.exists(csv_path):
        rospy.logwarn(f"No data.csv found in {folder_path}, skipping...")
        return False
    
    try:
        # Generate videos
        rospy.loginfo(f"Generating videos in {videos_folder} from {csv_path} ...")
        video_creator = VideoCreator(csv_path, videos_folder, fps=30, resolution=(576, 324))
        video_creator.create_videos()
        rospy.loginfo("Videos generated.")
        
        # Generate plot.html
        rospy.loginfo(f"Generating plot.html in {folder_path} ...")
        plot_arm_joints_and_xyz(
            data_csv=csv_path,
            actions_csv=actions_csv,
            x_axis="Frame Number",
            title=folder_path,
            logging_folder=folder_path
        )
        rospy.loginfo("plot.html generated.")
        return True
        
    except Exception as e:
        rospy.logerr(f"Error processing {folder_path}: {str(e)}")
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
    rospy.init_node('process_logged_folder_node')

    # Read ROS parameters
    logging_folder = rospy.get_param('LOGGING_FOLDER', None)
    logging_folder = "/home/stanford/DepthIndividual"
    logging_description = rospy.get_param('LOGGING_DESCRIPTION', None)
    # logging_description = "Test10agg"

    if logging_folder is None:
        rospy.logerr("LOGGING_FOLDER parameter must be set.")
        return

    # If logging_description is provided, process that specific folder
    if logging_description is not None:
        folder_path = os.path.join(logging_folder, logging_description)
        if os.path.exists(folder_path):
            process_demo_folder(folder_path)
        else:
            rospy.logerr(f"Folder not found: {folder_path}")
    else:
        # Otherwise, find and process all Demo* folders recursively
        rospy.loginfo(f"Searching for Demo* folders in {logging_folder}...")
        demo_folders = find_demo_folders(logging_folder)
        
        if not demo_folders:
            rospy.logwarn(f"No Demo* folders found in {logging_folder}")
            return
        
        rospy.loginfo(f"Found {len(demo_folders)} Demo* folders to process")
        
        success_count = 0
        for i, folder in enumerate(demo_folders, 1):
            rospy.loginfo(f"\n[{i}/{len(demo_folders)}] Processing: {folder}")
            if process_demo_folder(folder):
                success_count += 1
        
        rospy.loginfo(f"\nProcessing complete! Successfully processed {success_count}/{len(demo_folders)} folders.")

if __name__ == "__main__":
    main()