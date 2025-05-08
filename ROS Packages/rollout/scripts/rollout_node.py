#!/usr/bin/env python

import rospy
import argparse
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import pickle
from AdaptACT.procedures.controller import AutonomousController
from typing import Tuple


class RolloutController:
    def __init__(self, config_dict):
        """
        ROS Node for controlling the robot using the AutonomousController.
        """
        # Load parameters from argparse arguments
        self.train_dir = config_dict["train_dir"]
        self.ckpt_strategy = config_dict["ckpt_strategy"]
        self.rollout_len = config_dict["rollout_len"]
        self.device = config_dict["device"]
        self.node_name = config_dict["node_name"]

        rospy.init_node(self.node_name, anonymous=True)

        # self.safety_threshold = args.safety_threshold
        
        with open(f"{self.train_dir}/dataset_stats.pkl", "rb") as f:
            stats = pickle.load(f)

        # Initialize the AutonomousController
        self.controller = AutonomousController.from_train_dir(
            train_dir=self.train_dir,
            ckpt_strategy=self.ckpt_strategy
        )
 

    #     # ROS publishers and subscribers
    #     self.action_pub = rospy.Publisher("/robot/cmd_vel", Twist, queue_size=10)
    #     self.kill_sub = rospy.Subscriber("/robot/kill_signal", Bool, self.kill_callback)
    #     self.left_cam_sub = rospy.Subscriber("/camera/left/image_raw", Image, self.left_cam_callback)
    #     self.right_cam_sub = rospy.Subscriber("/camera/right/image_raw", Image, self.right_cam_callback)

    #     # Internal state
    #     self.bridge = CvBridge()
    #     self.left_image = None
    #     self.right_image = None
    #     self.kill_signal = False

    # def left_cam_callback(self, msg: Image):
    #     """Callback for the left camera image."""
    #     self.left_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    # def right_cam_callback(self, msg: Image):
    #     """Callback for the right camera image."""
    #     self.right_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    # def kill_callback(self, msg: Bool):
    #     """Callback for the kill signal."""
    #     self.kill_signal = msg.data
    #     if self.kill_signal:
    #         rospy.logwarn("Kill signal received! Stopping the robot.")
    #         self.stop_robot()

    # def stop_robot(self):
    #     """Stop the robot by publishing zero velocity."""
    #     twist = Twist()
    #     twist.linear.x = 0.0
    #     twist.angular.z = 0.0
    #     self.action_pub.publish(twist)

    # def get_observations(self) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Get observations from the robot's cameras.

    #     Returns:
    #         Tuple of left and right camera images as numpy arrays.
    #     """
    #     if self.left_image is None or self.right_image is None:
    #         raise RuntimeError("Camera images are not available yet.")
    #     return self.left_image, self.right_image

    # def safety_check(self, actions: np.ndarray) -> np.ndarray:
    #     """
    #     Apply safety guardrails to the actions.

    #     Args:
    #         actions: The raw actions predicted by the controller.

    #     Returns:
    #         Safe actions after applying guardrails.
    #     """
    #     actions = np.clip(actions, -self.safety_threshold, self.safety_threshold)
    #     return actions

    # def run(self):
    #     """Main loop for the ROS node."""
    #     rate = rospy.Rate(10)  # 10 Hz
    #     while not rospy.is_shutdown():
    #         if self.kill_signal:
    #             self.stop_robot()
    #             continue

    #         try:
    #             # Get observations
    #             left_img, right_img = self.get_observations()
    #             imgs = np.stack([left_img, right_img], axis=0)  # Shape: 2 x H x W x C
    #             qpos = np.zeros((self.controller._cfg.MODEL.STATE_DIM,))  # Placeholder for qpos

    #             # Get actions from the controller
    #             actions = self.controller.step(imgs, qpos)

    #             # Apply safety checks
    #             safe_actions = self.safety_check(actions)

    #             # Publish actions
    #             twist = Twist()
    #             twist.linear.x = safe_actions[0]
    #             twist.angular.z = safe_actions[1]
    #             self.action_pub.publish(twist)

    #         except RuntimeError as e:
    #             rospy.logerr(f"Error during control loop: {e}")
    #             self.stop_robot()

    #         rate.sleep()


if __name__ == "__main__":
    rospy.init_node("rollout_node", anonymous=True)
    parser = argparse.ArgumentParser(description="Rollout Node for AutonomousController")

    default_train_dir = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Models/trained_on_single_human_demos/Joint Control/20250426-190841_3795111b-3ebe-47e3-a4ad-345fdc5e0f41_train"
    parser.add_argument("--train_dir", type=str, default=default_train_dir, help="Path to the training directory")
    parser.add_argument("--ckpt_strategy", type=str, default="best", help="Checkpoint strategy: 'best', 'last', or 'none'")
    parser.add_argument("--rollout_len", type=int, default=400, help="Rollout length")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use: 'cpu' or 'cuda:X'")
    parser.add_argument("--safety_threshold", type=float, default=1.0, help="Safety threshold for action values")

    args = parser.parse_args()

    config_dict = {
        "train_dir": args.train_dir,
        "ckpt_strategy": args.ckpt_strategy,
        "rollout_len": args.rollout_len,
        "device": args.device,
        "safety_threshold": args.safety_threshold,
        "node_name": "rollout_node",
    }
    rollout_controller = RolloutController(config_dict)