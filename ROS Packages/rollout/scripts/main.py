#!/usr/bin/env python

import rospy
import argparse
import sys
import os
import numpy as np

from sensor_msgs.msg import Image,JointState
from geometry_msgs.msg import PoseStamped

# Add the folder containing the scripts to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../data_collection/scripts"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../rollout/scripts"))

from rollout_node import RolloutController
from initialize_exp import ExperimentInitializer
from csv_generator import MessageSynchronizer


def main():
    parser = argparse.ArgumentParser(description="Master Script for Surgical Robot Operations")

    # Add arguments for the master script
    parser.add_argument("--initialize", "-i", action="store_true", help="Initialize the experiment")
    parser.add_argument("--log", action="store_true", help="Generate CSV logs")
    parser.add_argument("--rollout", action="store_true", help="Perform rollout using the AutonomousController")

    # Arguments for initialization
    # parser.add_argument("--transform_lookup_wait_time", type=float, default=1.0, help="Transform lookup wait time")
    # parser.add_argument("--sleep_time_between_moves", type=float, default=0.7, help="Sleep time between moves")
    parser.add_argument("--ros_freq", type=float, default=10.0, help="ROS frequency")
    parser.add_argument("--reposition_ecm", action="store_true", help="Reposition ECM during initialization")
    # parser.add_argument("--position_diff_threshold", type=float, default=0.04, help="Position difference threshold")

    # Arguments for CSV generation
    parser.add_argument("--logging_description", type=str, help="Description of the data collection")
    # parser.add_argument("--logging_folder", type=str, default="/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Initial Samples/", help="Logging folder")
    parser.add_argument("--duration", type=int, default=15, help="Duration of the experiment in seconds")

    # Arguments for rollout
    parser.add_argument("--train_dir", type=str, required=False, help="Path to the training directory")
    parser.add_argument("--ckpt_strategy", type=str, default="best", help="Checkpoint strategy: 'best', 'last', or 'none'")
    parser.add_argument("--rollout_len", type=int, default=800, help="Rollout length")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use: 'cpu' or 'cuda:X'")
    parser.add_argument("--step_frequency", type=int, default=30, help="Frequency of steps in Hz")
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Initialize CRTK RAL
    ral = crtk.ral('master_script')

    # Configuration dictionaries
    initialize_config = {
        "parent_frames": ["Cart", "ECM_ref", "ECM_ref", "ECM_ref"],
        "child_frames": ["ECM_ref", "PSM1_ref", "PSM2_ref", "PSM3_ref"],
        "arm_names": ["ECM", "PSM1", "PSM2", "PSM3"],
        "transform_lookup_wait_time": 1.0,
        "sleep_time_between_moves": 0.7,
        "ros_freq": args.ros_freq,
        "reposition_ecm": args.reposition_ecm,
        "position_diff_threshold": 0.04
    }



    rollout_config = {
        "train_dir": args.train_dir,
        "ckpt_strategy": args.ckpt_strategy,
        "rollout_len": args.rollout_len,
        "device": args.device,
        "arm_names": ["PSM1", "PSM2"],
        "node_name": "rollout_node",
        "image_size": (324, 576),
        "step_frequency": args.step_frequency,
        "guardrail_thresholds": np.array([0.5, 0.4, 0.4, 1.0, 0.4, 0.4, 1.2]),
        "debug_mode": args.debug_mode
    }

    # Execute the selected functionality
    if args.initialize:
        rospy.loginfo("Initializing the experiment...")
        initializer = ExperimentInitializer(ral, initialize_config)
        success = initializer.run()
        if success:
            rospy.loginfo("Experiment initialization completed successfully.")
        else:
            rospy.logerr("Experiment initialization failed.")
        return


    if args.rollout:
        rospy.loginfo("Starting rollout...")
        rollout_controller = RolloutController(ral, rollout_config)
        rollout_controller.run()
        return

    rospy.logerr("No valid option selected. Use --initialize, --log, or --rollout.")
    parser.print_help()


if __name__ == "__main__":
    # main()
    print("Master script completed.")