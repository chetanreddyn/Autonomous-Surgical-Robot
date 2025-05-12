#!/usr/bin/env python

import rospy
import argparse
import numpy as np
from sensor_msgs.msg import Image, JointState
import cv2
from cv_bridge import CvBridge
from AdaptACT.procedures.controller import AutonomousController
import dvrk
import crtk
import pickle
import torch
import numpy as np
from message_filters import Subscriber, ApproximateTimeSynchronizer


class RolloutController:
    def __init__(self, ral, config_dict):
        """
        ROS Node for controlling the robot using the AutonomousController.
        """
        # Load parameters from argparse arguments
        self.train_dir = config_dict["train_dir"]
        self.ckpt_strategy = config_dict["ckpt_strategy"]
        self.rollout_len = config_dict["rollout_len"]
        self.device = config_dict["device"]
        self.arm_names = config_dict["arm_names"]
        self.image_size = config_dict["image_size"]
        self.step_frequency = config_dict["step_frequency"]
        self.guardrail_thresholds = config_dict["guardrail_thresholds"]
        self.debug_mode = config_dict["debug_mode"]
        self.loginfo = config_dict["loginfo"]
        # Initialize dVRK arm objects
        self.arm_objs = self.initialize_arm_objs(ral)  # Will be used to move the arms later

        # Initialize the AutonomousController
        self.controller = AutonomousController.from_train_dir(
            train_dir=self.train_dir,
            ckpt_strategy=self.ckpt_strategy,
            rollout_len=self.rollout_len,
            device=self.device
        )

        # Initialize CvBridge for image processing
        self.bridge = CvBridge()

        # Initialize placeholders for synchronized data
        self.camera_right_image = None
        self.camera_left_image = None
        self.joint_positions = {arm_name: None for arm_name in self.arm_names}
        self.jaw_angles = {arm_name: None for arm_name in self.arm_names}

        # Create subscribers for synchronized topics
        self.camera_right_sub = Subscriber("/camera_right/image_raw", Image)
        self.camera_left_sub = Subscriber("/camera_left/image_raw", Image)
        self.joint_state_subs = [
            Subscriber(f"/{arm_name}/setpoint_js", JointState) for arm_name in self.arm_names
        ]
        self.jaw_angle_subs = [
            Subscriber(f"/{arm_name}/jaw/setpoint_js", JointState) for arm_name in self.arm_names
        ]

        # Synchronize all topics
        self.sync = ApproximateTimeSynchronizer(
            [self.camera_right_sub, self.camera_left_sub] + self.joint_state_subs + self.jaw_angle_subs,
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.synchronized_callback)

    def initialize_arm_objs(self, ral):
        """
        Initialize dVRK arm objects.
        """
        arm_objs = {arm_name: dvrk.psm(ral, arm_name) for arm_name in self.arm_names}
        return arm_objs

    def synchronized_callback(self, camera_right_msg, camera_left_msg, *joint_and_jaw_msgs):
        """
        Callback to process synchronized messages from all topics.
        """
        try:
            # Process camera images
            self.camera_right_image = cv2.resize(self.bridge.imgmsg_to_cv2(camera_right_msg, desired_encoding='bgr8'), self.image_size)

            self.camera_left_image = cv2.resize(self.bridge.imgmsg_to_cv2(camera_left_msg, desired_encoding='bgr8'), self.image_size)

            # Process joint states and jaw angles
            num_arms = len(self.arm_names)
            for i, arm_name in enumerate(self.arm_names):
                self.joint_positions[arm_name] = np.array(joint_and_jaw_msgs[i].position)
                self.jaw_angles[arm_name] = np.array(joint_and_jaw_msgs[num_arms + i].position)

        except Exception as e:
            rospy.logerr(f"Failed to process synchronized messages: {e}")

    def get_from_robot(self):
        """
        Returns the latest synchronized images and joint positions from the robot.

        :return: Tuple (images, qpos)
                 - images: A NumPy array of shape (N, H, W, C) where N is the number of cameras.
                 - qpos: A NumPy array of shape (D,) containing the joint positions and jaw angles.
        """
        # Ensure images are available
        if self.camera_right_image is None or self.camera_left_image is None:
            rospy.logwarn("Images from cameras are not yet available.")
            return None, None

        # Ensure joint positions and jaw angles are available
        if any(pos is None for pos in self.joint_positions.values()):
            rospy.logwarn("Joint positions for some arms are not yet available.")
            return None, None

        if any(angle is None for angle in self.jaw_angles.values()):
            rospy.logwarn("Jaw angles for some arms are not yet available.")
            return None, None

        # Stack images into a single NumPy array with shape (N, H, W, C)
        try:
            images = np.stack([self.camera_right_image, self.camera_left_image], axis=0)
        except Exception as e:
            print(self.camera_right_image.shape, self.camera_left_image.shape)
        # Concatenate joint positions and jaw angles from all arms into a single NumPy array
        qpos = np.concatenate([
            np.concatenate([self.joint_positions[arm_name], self.jaw_angles[arm_name]])
            for arm_name in self.arm_names
        ])
        images = images.astype(np.float32) / 255.0

        return images, qpos

    def process_action(self, action):
        """
        Process the action from the controller and send it to the robot.

        :param action: The action to be sent to the robot.
        """
        # Example of how to process the action
        for i, arm_name in enumerate(self.arm_names):
            joint_positions = action[7*i: 7*(i + 1)]

            measured_joint_state = np.array(self.arm_objs[arm_name].measured_js()[0])
            measured_jaw_state = np.array(self.arm_objs[arm_name].jaw.measured_js()[0])
            measured_joint_state = np.concatenate([measured_joint_state, measured_jaw_state])

            target_joint_state = joint_positions
            # print(initial_joint_state.round(2),target_joint_state.round(2))
            diff = np.abs(measured_joint_state - target_joint_state)
            # rospy.loginfo(str(measured_joint_state[3].round(2)) + arm_name)


            if np.any(diff > self.guardrail_thresholds):
                diff_joints = np.where(diff > self.guardrail_thresholds)[0]
                # rospy.logfatal(f"Joint state discrepancy for {arm_name} is too large at joint {diff_joints} | diff : {np.round(diff,2)}")
                continue

            if not self.debug_mode:
                self.arm_objs[arm_name].move_jp(joint_positions[:-1])
                self.arm_objs[arm_name].jaw.move_jp(np.array([joint_positions[-1]]))


    def run(self):
        """
        Main loop to perform rollout using the AutonomousController.
        """
        rospy.loginfo("Starting rollout...")
        self.controller.reset()
        rate = rospy.Rate(self.step_frequency)
        step = 0
        t0 = rospy.get_time()
        while not rospy.is_shutdown() and step < self.rollout_len:
            # Get images and joint positions from the robot
            images, qpos = self.get_from_robot()
            if images is None or qpos is None:
                rospy.sleep(0.1)
                continue

            # Perform a step of the autonomous controller
            try:
                action = self.controller.step(images, qpos)
                self.process_action(action)
            except RuntimeError as e:
                rospy.logerr(f"Error during rollout: {e}")
                break


            rate.sleep()
            step += 1
            ti = rospy.get_time()
            time_stamp = ti - t0

            if self.loginfo:
                rospy.loginfo(f"Time: {time_stamp:.2f} | Step {step}/{self.rollout_len} completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rollout Node for AutonomousController")

    default_train_dir = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Models/trained_on_single_human_demos/Joint Control/20250503-191543_masterful-rat_train"
    parser.add_argument("--train_dir", type=str, default=default_train_dir, help="Path to the training directory")
    parser.add_argument("--ckpt_strategy", type=str, default="best", help="Checkpoint strategy: 'best', 'last', or 'none'")
    parser.add_argument("-T", "--rollout_len", type=int, default=800, help="Rollout length")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use: 'cpu' or 'cuda:X'")
    parser.add_argument("--step_frequency", type=int, default=30, help="Frequency of steps in Hz")
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode")
    parser.add_argument("--loginfo", action="store_true", help="Enable loginfo mode")

    args, unknown = parser.parse_known_args()
    rospy.sleep(1)
    config_dict = {
        "train_dir": args.train_dir,
        "ckpt_strategy": args.ckpt_strategy,
        "rollout_len": args.rollout_len,
        "device": args.device,
        "arm_names": ["PSM1", "PSM2"],
        "node_name": "rollout_node",
        "image_size": (324, 576),
        "step_frequency": args.step_frequency,
        "guardrail_thresholds": np.array([0.5, 0.4, 0.4, 1.0, 0.4, 0.4, 1.2]),
        "debug_mode": args.debug_mode,
        "loginfo": args.loginfo
    }
    ral = crtk.ral('RolloutNode')

    rollout_controller = RolloutController(ral, config_dict)
    rollout_controller.run()