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
from datetime import datetime
import os
import shutil
import sys
import csv
from message_filters import Subscriber, ApproximateTimeSynchronizer


class RolloutController:
    def __init__(self, ral, config_dict):
        """
        ROS Node for controlling the robot using the AutonomousController.
        """
        # Load parameters from argparse arguments
        self.train_dir = config_dict["train_dir"]
        self.logging_folder = config_dict["logging_folder"]
        self.logging_description = config_dict["logging_description"]
        self.ckpt_strategy = config_dict["ckpt_strategy"]
        self.ckpt_path = config_dict["ckpt_path"]
        self.rollout_len = config_dict["rollout_len"]
        self.device = config_dict["device"]
        self.arm_names = config_dict["arm_names"]
        self.image_size = config_dict["image_size"]
        self.step_frequency = config_dict["step_frequency"]
        self.guardrail_thresholds = config_dict["guardrail_thresholds"]
        self.debug_mode = config_dict["debug_mode"]
        self.loginfo = config_dict["loginfo"]
        self.log_actions = config_dict["log_actions"]
        self.time_format = "%Y-%m-%d %H:%M:%S.%f"
        self.automated_arms = config_dict["automated_arms"]
        self.record = config_dict["record"]

        # Initialize dVRK arm objects
        self.arm_objs = self.initialize_arm_objs(ral)  # Will be used to move the arms later

        # Initialize the AutonomousController
        self.controller = AutonomousController.from_train_dir(
            train_dir=self.train_dir,
            ckpt_strategy=self.ckpt_strategy,
            ckpt_path = self.ckpt_path,
            rollout_len=self.rollout_len,
            device=self.device
        )

        rospy.loginfo("Model Loaded")
        # Initialize CvBridge for image processing
        self.bridge = CvBridge()

        # Initialize placeholders for synchronized data
        self.camera_right_image = None
        self.camera_left_image = None
        self.joint_positions = {arm_name: None for arm_name in self.arm_names}
        self.jaw_angles = {arm_name: None for arm_name in self.arm_names}

        # # Create subscribers for synchronized topics
        # self.camera_right_sub = Subscriber("/camera_right/image_raw", Image)
        # self.camera_left_sub = Subscriber("/camera_left/image_raw", Image)
        # self.joint_state_subs = [
        #     Subscriber(f"/{arm_name}/setpoint_js", JointState) for arm_name in self.arm_names
        # ]
        # self.jaw_angle_subs = [
        #     Subscriber(f"/{arm_name}/jaw/setpoint_js", JointState) for arm_name in self.arm_names
        # ]

        # # Synchronize all topics
        # self.sync = ApproximateTimeSynchronizer(
        #     [self.camera_right_sub, self.camera_left_sub] + self.joint_state_subs + self.jaw_angle_subs,
        #     queue_size=10,
        #     slop=0.1
        # )
        # self.sync.registerCallback(self.synchronized_callback)

        # Create individual subscribers for each camera
        rospy.Subscriber("/camera_right/image_raw", Image, self.camera_right_callback)
        rospy.Subscriber("/camera_left/image_raw", Image, self.camera_left_callback)

        # Create individual subscribers for each arm's joint state and jaw angle
        for arm_name in self.arm_names:
            rospy.Subscriber(f"/{arm_name}/setpoint_js", JointState, self.joint_state_callback, callback_args=arm_name)
            rospy.Subscriber(f"/{arm_name}/jaw/setpoint_js", JointState, self.jaw_angle_callback, callback_args=arm_name)

    def camera_right_callback(self, msg):
        """
        Callback for the right camera image.
        """
        try:
            self.camera_right_image = cv2.resize(self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'), self.image_size)
            # rospy.loginfo("Received Right Image")
        except Exception as e:
            rospy.logerr(f"Failed to process right camera image: {e}")

    def camera_left_callback(self, msg):
        """
        Callback for the left camera image.
        """
        try:
            self.camera_left_image = cv2.resize(self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'), self.image_size)
            # rospy.loginfo("Received Left Image")
        except Exception as e:
            rospy.logerr(f"Failed to process left camera image: {e}")

    def joint_state_callback(self, msg, arm_name):
        """
        Callback for joint state messages.
        """
        try:
            self.joint_positions[arm_name] = np.array(msg.position)
            # rospy.loginfo("Received Joint State")
        except Exception as e:
            rospy.logerr(f"Failed to process joint state for {arm_name}: {e}")

    def jaw_angle_callback(self, msg, arm_name):
        """
        Callback for jaw angle messages.
        """
        try:
            self.jaw_angles[arm_name] = np.array(msg.position)
        except Exception as e:
            rospy.logerr(f"Failed to process jaw angle for {arm_name}: {e}")

    def initialize_arm_objs(self, ral):
        """
        Initialize dVRK arm objects.
        """
        arm_objs = {arm_name: dvrk.psm(ral, arm_name) for arm_name in self.arm_names}
        return arm_objs

    # def synchronized_callback(self, camera_right_msg, camera_left_msg, *joint_and_jaw_msgs):
    #     """
    #     Callback to process synchronized messages from all topics.
    #     """
    #     try:
    #         # Process camera images
    #         self.camera_right_image = cv2.resize(self.bridge.imgmsg_to_cv2(camera_right_msg, desired_encoding='bgr8'), self.image_size)

    #         self.camera_left_image = cv2.resize(self.bridge.imgmsg_to_cv2(camera_left_msg, desired_encoding='bgr8'), self.image_size)

    #         # Process joint states and jaw angles
    #         num_arms = len(self.arm_names)
    #         for i, arm_name in enumerate(self.arm_names):
    #             self.joint_positions[arm_name] = np.array(joint_and_jaw_msgs[i].position)
    #             self.jaw_angles[arm_name] = np.array(joint_and_jaw_msgs[num_arms + i].position)

    #     except Exception as e:
    #         rospy.logerr(f"Failed to process synchronized messages: {e}")

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
            if arm_name in self.automated_arms:
                if len(action)!=7*len(self.arm_names):
                    rospy.logerr(f"Length of model output must be 7*{len(self.arm_names)} i.e 7*num_arms but is {len(action)}")
                    continue

                joint_positions = action[7*i: 7*(i + 1)] # Assumes the order PSM1, PSM2, PSM3

                measured_joint_state = np.array(self.arm_objs[arm_name].measured_js()[0])
                measured_jaw_state = np.array(self.arm_objs[arm_name].jaw.measured_js()[0])
                measured_joint_state = np.concatenate([measured_joint_state, measured_jaw_state])

                target_joint_state = joint_positions
                # print(initial_joint_state.round(2),target_joint_state.round(2))
                diff = np.abs(measured_joint_state - target_joint_state)
                # rospy.loginfo(str(measured_joint_state[3].round(2)) + arm_name)


                if np.any(diff > self.guardrail_thresholds):
                    diff_joints = np.where(diff > self.guardrail_thresholds)[0]
                    rospy.logwarn(f"Joint state discrepancy for {arm_name} is too large at joint {diff_joints} | diff : {np.round(diff,2)}")
                    # continue

                if not self.debug_mode:
                    self.arm_objs[arm_name].move_jp(joint_positions[:-1])
                    self.arm_objs[arm_name].jaw.move_jp(np.array([joint_positions[-1]]))

    def generate_csv_columns(self):
        columns = ["Epoch Time", "Time (Seconds)", "Frame Number"]
        for arm_name in self.arm_names:
            # Joints and jaw
            for i in range(1, 7):
                columns.append(f"{arm_name}_joint_{i}")
            columns.append(f"{arm_name}_jaw")

        return columns

    def process_timestamp(self, time_stamp):
        '''
        Processes the time stamp of the message
        '''
        message_time = datetime.fromtimestamp(time_stamp.to_sec())
        epoch_time_formatted = message_time.strftime(self.time_format)
        return epoch_time_formatted

    def run(self):
        """
        Main loop to perform rollout using the AutonomousController.
        """

        rospy.loginfo("Starting rollout...")
        self.controller.reset()
        rate = rospy.Rate(self.step_frequency)
        step = 0

        if self.record:
            rospy.set_param("rollout_started", True)

            rospy.loginfo("Waiting for Recording to start...")
            recording_started = False
            while not recording_started and not rospy.is_shutdown():
                recording_started = rospy.get_param("recording_started", False) # Default value is False
                # rospy.loginfo("Waiting for recording to start...")
                if not recording_started:
                    rospy.sleep(0.01)
            rospy.sleep(0.5)


        if self.log_actions:
            csv_path = os.path.join(self.logging_folder, self.logging_description, "rollout_actions.csv")
            columns = self.generate_csv_columns()
            csv_file = open(csv_path, mode='w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(columns)

        t0 = rospy.get_time()

        while not rospy.is_shutdown() and step < self.rollout_len:
            # Get images and joint positions from the robot
            images, qpos = self.get_from_robot()
            if images is None or qpos is None:
                rospy.sleep(0.01)
                rospy.logwarn("images or qpos is none")
                continue
            # rospy.loginfo(f"images shape: {images.shape} | qpos shape: {qpos.shape}")

            # Perform a step of the autonomous controller
            action = self.controller.step(images, qpos)
            del images
            self.process_action(action)


            rate.sleep()
            step += 1
            ti = rospy.get_time()
            time_stamp = ti - t0

            if self.log_actions:
                epoch_time_formatted = self.process_timestamp(rospy.Time.now())
                row = [epoch_time_formatted, time_stamp, step]

                row.extend(action.tolist())
                csv_writer.writerow(row)
                
            if self.loginfo:
                rospy.loginfo(f"Time: {time_stamp:.2f} | Step {step}/{self.rollout_len} completed.")

            torch.cuda.empty_cache()
            rospy.loginfo(torch.cuda.max_memory_allocated())

        if self.log_actions:
            csv_file.close()
            rospy.loginfo(f"Rollout actions logged to {csv_path}")

if __name__ == "__main__":
    ral = crtk.ral('RolloutNode')

    # TRAIN_DIR = rospy.get_param("TRAIN_DIR")
    # TRAIN_DIR = rospy.get_param("TRAIN_DIR", "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Models/4_merged_training/Joint Control/20250516-130148_original-seal_train")
    TRAIN_DIR = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Models/4_merged_training/Joint Control/20250516-130148_original-seal_train"
    LOGGING_FOLDER = rospy.get_param("LOGGING_FOLDER", default="/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Rollouts Collaborative")
    LOGGING_DESCRIPTION = rospy.get_param("LOGGING_DESCRIPTION", default="Test")
    parser = argparse.ArgumentParser(description="Rollout Node for AutonomousController")
    parser.add_argument("--ckpt_strategy", type=str, default="best", help="Checkpoint strategy: 'best', 'last', or 'none'")
    parser.add_argument("-N", "--rollout_len", type=int, default=450, help="Rollout length")
    parser.add_argument("--step_frequency", type=int, default=30, help="Frequency of steps in Hz")
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode")
    parser.add_argument("--loginfo", action="store_true", help="Enable loginfo mode")
    parser.add_argument("--log_actions", action="store_true", help="Enable logging of actions")
    parser.add_argument("-a1", default="", help = "Arm1 to Automate")
    parser.add_argument("-a2", default="", help = "Arm2 to Automate")
    parser.add_argument("--record", action="store_true", help="Record the rollout")
    
    args, unknown = parser.parse_known_args()

    arm_names = ["PSM1", "PSM2", "PSM3"]

    if args.a1 in arm_names and args.a2 in arm_names:
        automated_arms = [args.a1, args.a2]

    elif args.a1 in arm_names and args.a2 not in arm_names:
        automated_arms = [args.a1]

    elif args.a1 not in arm_names and args.a2 in arm_names:
        automated_arms = [args.a2]

    else:
        automated_arms = ["PSM1", "PSM2"]


    for arm in automated_arms:
        if arm not in ["PSM1", "PSM2", "PSM3"]:
            print(f"Invalid arm name: {arm}. Valid names are: PSM1, PSM2, PSM3")
            sys.exit(1)


    config_dict = {
        "train_dir": TRAIN_DIR,
        "logging_folder": LOGGING_FOLDER,
        "logging_description": LOGGING_DESCRIPTION,
        "ckpt_strategy": args.ckpt_strategy,
        "ckpt_path": os.path.join(TRAIN_DIR, "policy_epoch_20000_seed_0.ckpt"),
        # "ckpt_path": os.path.join(args.train_dir, "policy_best_13933.ckpt"),
        "rollout_len": args.rollout_len,
        "device": "cuda:0",
        "arm_names": ["PSM1", "PSM2"],
        "node_name": "rollout_node",
        "image_size": (324, 576),
        "step_frequency": args.step_frequency,
        "guardrail_thresholds": np.array([0.5, 0.4, 0.4, 1.2, 0.6, 0.4, 2.5]),
        "debug_mode": args.debug_mode,
        "loginfo": args.loginfo,
        "log_actions": args.log_actions,
        "automated_arms": automated_arms,
        "record": args.record
    }


    rollout_controller = RolloutController(ral, config_dict)
    rollout_controller.run()




    