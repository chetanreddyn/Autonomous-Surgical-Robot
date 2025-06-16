#!/usr/bin/env python

import os
import rospy

from frames_to_video import VideoCreator
from plotter import plot_arm_joints_and_xyz

def main():
    rospy.init_node('process_logged_folder_node')

    # Read ROS parameters
    logging_folder = rospy.get_param('LOGGING_FOLDER', None)
    # logging_folder = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Rollouts/Rollouts Three Handed Initial/Two Arms Autonomous"
    logging_description = rospy.get_param('LOGGING_DESCRIPTION', None)
    # logging_description = "Test10agg"

    if logging_folder is None or logging_description is None:
        rospy.logerr("LOGGING_FOLDER and LOGGING_DESCRIPTION parameters must be set.")
        return

    folder_path = os.path.join(logging_folder, logging_description)
    csv_path = os.path.join(folder_path, "data.csv")
    actions_csv = os.path.join(folder_path, "rollout_actions.csv")
    videos_folder = os.path.join(folder_path, "videos")

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

if __name__ == "__main__":
    main()