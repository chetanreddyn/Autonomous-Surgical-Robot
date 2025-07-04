#!/usr/bin/env python

import rospy
import tf2_ros
import PyKDL
import dvrk
import crtk
from sensor_msgs.msg import JointState
import numpy as np
import argparse


class ExperimentInitializer:
    def __init__(self, ral, config_dict):
        """
        Initialize the ExperimentInitializer object.

        :param config_dict: Dictionary containing configuration parameters.
        """

        self.ros_freq = config_dict['ros_freq']
        self.reposition_ecm = config_dict['reposition_ecm']
        self.position_diff_threshold = config_dict['position_diff_threshold']

        # Initialize dVRK arms
        # self.ecm_name = config_dict['arm_names'][0]
        # self.arm1_name = config_dict['arm_names'][1]
        # self.arm2_name = config_dict['arm_names'][2]
        # self.arm3_name = config_dict['arm_names'][3]

        # self.num_transforms = len(config_dict['parent_frames'])
        # self.parent_frames = config_dict['parent_frames']
        # self.child_frames = config_dict['child_frames']
        self.arm_names = config_dict['arm_names']
        self.generate_frames_from_arm_names(self.arm_names)
        self.num_transforms = len(self.parent_frames)

        self.jaw_names = {arm_name: arm_name+"_jaw" for arm_name in self.arm_names}

        self.sleep_time_between_moves = config_dict['sleep_time_between_moves']

        self.transform_lookup_wait_time = config_dict['transform_lookup_wait_time']
        # self.arm1 = dvrk.psm(ral, self.arm1_name)
        # self.arm2 = dvrk.psm(ral, self.arm2_name)
        # self.arm3 = dvrk.psm(ral, self.arm3_name)
        # self.ecm = dvrk.ecm(ral, self.ecm_name) # Arm 3 is ECM

        # self.arm_objs = {self.arm1_name: self.arm1, self.arm2_name: self.arm2,
        #                  self.arm3_name: self.arm3, self.ecm_name: self.ecm}
        
        # self.arm_objs = {self.arm1_name: self.arm1, self.arm2_name: self.arm2, self.ecm_name: self.ecm}
        self.arm_objs = {}
        for arm_name in self.arm_names:
            if "ECM" in arm_name:
                self.arm_objs[arm_name] = dvrk.ecm(ral, arm_name)
            else:
                self.arm_objs[arm_name] = dvrk.psm(ral, arm_name)

        self.move_cp_goals_received = False
        self.move_cp_goals = None # PyKDL.Frame() type transforms to send to move_cp topic

        # Initialize tf2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.loginfo("Loading transforms...")
        self.loaded_transforms = self.load_transforms() # If successful, returns True and transforms are loaded into self.move_cp_goals
  
        # print(self.move_cp_goals)

        # Subscribe to jaw_angles_ref topic
        self.jaw_angles = {}
        self.loaded_jaw_angles = False

    def generate_frames_from_arm_names(self, arm_names):
        self.parent_frames = []
        self.child_frames = []

        for arm_name in arm_names:
            if arm_name == "ECM":
                self.parent_frames.append("Cart")
                self.child_frames.append("ECM_ref")
            else:
                self.parent_frames.append("ECM_ref")
                self.child_frames.append(arm_name+"_ref")

    def jaw_angles_callback(self, msg):
        """
        Callback to handle jaw angles from the jaw_angles_ref topic.

        :param msg: JointState message containing jaw angles.
        """
        if self.loaded_jaw_angles:
            return
        
        for jaw_name, position in zip(msg.name, msg.position):
            self.jaw_angles[jaw_name] = position
        rospy.loginfo("Received Jaw Angles")
        self.loaded_jaw_angles = True
        rospy.loginfo("Jaw angles loaded successfully.")

    def load_transforms(self):        
        """returns a boolean indicating if the transforms were loaded successfully"""
        self.move_cp_goals = {}
        for i in range(self.num_transforms):
            parent_frame = self.parent_frames[i]
            child_frame = self.child_frames[i]

            goal = None
            t0 = rospy.Time.now().to_sec()
            # print(t0)
            while not goal:
                goal = self.get_transform(parent_frame, child_frame)
                rospy.sleep(0.1)

                if rospy.Time.now().to_sec() - t0 > self.transform_lookup_wait_time:
                    rospy.logerr("Transform lookup timed out after {} seconds Looking for {} to {} Transform".format(
                        self.transform_lookup_wait_time, parent_frame, child_frame))
                    rospy.logerr("Consider increasing the transform_lookup_wait_time in the config_dict or check if the reference frames are being published correctly.")
                    
                    return False
               
            self.move_cp_goals[parent_frame+"_to_"+child_frame] = goal
        return True


    def get_transform(self, parent_frame, child_frame):
        """
        Get the transform between a parent and child frame using tf2.

        :param parent_frame: Parent frame name.
        :param child_frame: Child frame name.
        :return: PyKDL.Frame object representing the transform.
        """
        try:
            # Our reference frames are published as tf frames
            transform = self.tf_buffer.lookup_transform(parent_frame, child_frame, rospy.Time(0), rospy.Duration(1.0))
            translation = transform.transform.translation
            rotation = transform.transform.rotation

            # Create a PyKDL.Frame object for the transform
            goal = PyKDL.Frame()
            goal.p = PyKDL.Vector(translation.x, translation.y, translation.z)
            goal.M = PyKDL.Rotation.Quaternion(rotation.x, rotation.y, rotation.z, rotation.w)

            return goal
        except Exception as e:
            rospy.logerr(f"Failed to get transform from {parent_frame} to {child_frame}: {e}")
            return None

    def publish_transform(self, parent_frame, child_frame, arm_name):
        """
        Publish the transform from parent_frame to child_frame using move_cp.

        :param parent_frame: Parent frame name.
        :param child_frame: Child frame name.
        :param arm: dVRK arm object to move.
        """
        arm_obj = self.arm_objs[arm_name]
        


        goal = self.move_cp_goals.get(parent_frame+"_to_"+child_frame)
        if goal is None:
            rospy.logerr("Transform not found for {} to {}. Cannot publish.".format(parent_frame, child_frame))
            return
        
        else:

            # Safety Check
            current_position = arm_obj.setpoint_cp()
            error = (current_position.p - goal.p).Norm()
            if error> self.position_diff_threshold:
                rospy.logfatal(f"Current position is too far from the goal position for {arm_name} | Distance :{error:.4f}  | Aborting for Safety | Check the position_diff_threshold")
                rospy.signal_shutdown("Safety Check Failed")

            else:
                rospy.loginfo("Moving {}".format(arm_name))
                arm_obj.move_cp(goal).wait(True)

    def open_jaws(self):
        # open the jaws
        for arm_name in self.arm_names:
            if arm_name == "ECM":
                continue
            arm_obj = self.arm_objs[arm_name]
            jaw_name = self.jaw_names[arm_name]

            arm_obj.jaw.move_jp(np.array([np.pi/2]))
            rospy.loginfo(f"Opened {arm_name} jaw")
            rospy.sleep(self.sleep_time_between_moves)

            if rospy.is_shutdown():
                rospy.logerr("Interrupted. Shutting down experiment initializer")
                return False
            
        return True

    def initialize_arms(self):
        # Moves the arms in self.arm_names to the loaded reference transforms
        for i in range(self.num_transforms):
              
            parent_frame = self.parent_frames[i]
            child_frame = self.child_frames[i]
            arm_name = self.arm_names[i]
            if not self.reposition_ecm and arm_name == "ECM":
                rospy.logwarn("Skipping ECM repositioning because reposition_ecm is set to False")
                continue
            self.publish_transform(parent_frame,child_frame,arm_name)
            rospy.sleep(self.sleep_time_between_moves)
            if rospy.is_shutdown():
                rospy.logerr("Interrupted. Shutting down experiment initializer")
                return False
            
        return True

    def initialise_jaw_angles(self):
        rospy.Subscriber("jaw_angles_ref", JointState, self.jaw_angles_callback)

        while not self.loaded_jaw_angles and not rospy.is_shutdown():
            rospy.loginfo("Waiting for jaw angles to be loaded...")
            rospy.sleep(1/self.ros_freq)

        # Move the jaws to the loaded angles
        for arm_name in self.arm_names:
            if arm_name == "ECM":
                continue
            arm_obj = self.arm_objs[arm_name]
            jaw_name = self.jaw_names[arm_name]
            jaw_angle = self.jaw_angles[jaw_name]

            arm_obj.jaw.move_jp(np.array([jaw_angle]))
            rospy.loginfo(f"Moved {arm_name} jaw")
            rospy.sleep(self.sleep_time_between_moves)

            if rospy.is_shutdown():
                rospy.logerr("Interrupted. Shutting down experiment initializer")
                return False
            
        return True
            
    def run(self):
        """
        Run the experiment initialization process.
        """
        rospy.loginfo("Initializing experiment...")
        rospy.sleep(self.sleep_time_between_moves)
        # Publish transform from ECM_ref to PSM1_ref as ECM to PSM1


        
        if not self.loaded_transforms:
            rospy.logerr("Reference Transforms not loaded successfully. Cannot proceed.")
            return False

        success = self.open_jaws()
        if not success:
            return False
        
        success = self.initialize_arms()
        if not success:
            return False

        success = self.initialise_jaw_angles()
        if not success:
            return False

        rospy.loginfo("Experiment initialization complete.")
        return True


        


if __name__ == "__main__":
    # rospy.init_node("experiment_initializer", anonymous=True)

    # Configuration dictionary
    config_dict = {"arm_names": ["ECM", "PSM1", "PSM2", "PSM3"],
                   "transform_lookup_wait_time": 1.0,
                   "sleep_time_between_moves": 1,
                   "ros_freq": 10.0,
                   "reposition_ecm": True,
                   "position_diff_threshold": 0.08
    }

    ral = crtk.ral('experiment_initializer')
    # Create ExperimentInitializer object and run the initialization
    initializer = ExperimentInitializer(ral,config_dict)
    initializer.run()