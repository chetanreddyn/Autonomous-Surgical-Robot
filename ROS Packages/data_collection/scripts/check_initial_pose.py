#!/usr/bin/env python

'''
check_initial_pose.py
Used to check if the Arm base has moved and how far it is from the saved reference frames
'''

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
import argparse

class TransformChecker:
    def __init__(self, config_dict):
        """
        Initialize the TransformChecker object.

        :param config_dict: Dictionary containing configuration parameters.
        """
        self.parent_frames = config_dict['parent_frames']
        self.child_frames = config_dict['child_frames']
        self.ref_child_frames = config_dict['ref_child_frames']
        self.arm_names = config_dict['arm_names']
        self.ecm_name = config_dict['ecm_name']
        self.rate = rospy.Rate(config_dict['rospy_freq'])
        self.significant_digits = config_dict['significant_digits']
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # SUJ joint angles data
        self.suj_joint_angles = {}
        self.suj_joint_angles_ref = {}
        self.suj_joint_angles_topics = config_dict['suj_joint_angles_topics']
        self.suj_joint_angles_ref_topics = config_dict['suj_joint_angles_ref_topics']

    def suj_joint_callback(self, msg, info):
        """
        Callback function to store SUJ joint angles.

        :param msg: JointState message containing SUJ joint angles.
        :param data_type: "current" for current angles, "ref" for reference angles.
        """
        # rospy.loginfo(f"Received {info['topic_type']} joint angles for {info['arm_name']}: {msg.position}")
        if info['topic_type'] == "current":
            self.suj_joint_angles[info['arm_name']] = msg.position
        elif info['topic_type'] == "ref":
            self.suj_joint_angles_ref[info['arm_name']] = msg.position

    def get_transform(self, parent_frame, child_frame):
        """
        Get the transform between a parent and child frame using tf2.

        :param parent_frame: Parent frame name.
        :param child_frame: Child frame name.
        :return: TransformStamped object representing the transform, or None if not found.
        """
        try:
            return self.tf_buffer.lookup_transform(parent_frame, child_frame, rospy.Time(0), rospy.Duration(1.0))
        except Exception as e:
            rospy.logerr(f"Failed to get transform from {parent_frame} to {child_frame}: {e}")
            return None

    def calculate_error(self, transform1, transform2):
        """
        Calculate the error metric between two transforms.

        :param transform1: First transform as a TransformStamped.
        :param transform2: Second transform as a TransformStamped.
        :return: Error metric (Euclidean distance between translations).
        """
        # Calculate translation error
        dx = transform1.transform.translation.x - transform2.transform.translation.x
        dy = transform1.transform.translation.y - transform2.transform.translation.y
        dz = transform1.transform.translation.z - transform2.transform.translation.z
        translation_error = (dx**2 + dy**2 + dz**2)**0.5

        # Calculate rotation error (difference in quaternion components)
        q1 = transform1.transform.rotation
        q2 = transform2.transform.rotation
        rotation_error = abs(q1.x - q2.x) + abs(q1.y - q2.y) + abs(q1.z - q2.z) + abs(q1.w - q2.w)

        return translation_error, rotation_error

    def check_transforms(self):
        """
        Check the transforms for each parent-child pair and compare with the corresponding _ref frame.
        """
        while not rospy.is_shutdown():
            for parent_frame, child_frame,ref_child_frame in zip(self.parent_frames, self.child_frames,self.ref_child_frames):
                # Get the transform for the child frame
                child_transform = self.get_transform(parent_frame, child_frame)
                child_ref_transform = self.get_transform(parent_frame, ref_child_frame)


                if child_transform is None:
                    rospy.logwarn(f"Transform for {parent_frame} to {child_frame} not found.")
                    continue

                # Get the transform for the child_ref frame
                if child_ref_transform is None:
                    rospy.logwarn(f"Transform for {parent_frame} to {ref_child_frame} not found.")
                    continue

                # Calculate the error metric
                translation_error, rotation_error = self.calculate_error(child_transform, child_ref_transform)
                print(f"{child_frame:<5}: ({translation_error:.3f},{rotation_error:.3f})", end=" | ")
            print()
            self.rate.sleep()



    def create_suj_subscribers(self):

        # Subscribe to SUJ joint topics
        for arm_name,topic in self.suj_joint_angles_topics.items():
            rospy.Subscriber(topic, JointState, self.suj_joint_callback, 
                             callback_args={"topic_type": "current", "arm_name": arm_name})

        for arm_name,topic in self.suj_joint_angles_ref_topics.items():
            rospy.Subscriber(topic, JointState, self.suj_joint_callback, 
                             callback_args={"topic_type": "ref", "arm_name": arm_name})

    def check_suj_joint_angles(self):
        """
        Check the SUJ joint angles for each arm and calculate errors.
        """
        # Create subscribers for SUJ joint angles
        self.create_suj_subscribers()

        while not rospy.is_shutdown():

            if not self.suj_joint_angles or not self.suj_joint_angles_ref:
                rospy.logwarn("SUJ joint angles or reference angles not received yet.")
                self.rate.sleep()
                continue

            for joint_name, current_angles in self.suj_joint_angles.items():
                if joint_name in self.suj_joint_angles_ref:
                    ref_angles = self.suj_joint_angles_ref[joint_name]
                    errors = [round((current - ref),self.significant_digits)
                            for current, ref in zip(current_angles, ref_angles)]
                    print(f"{joint_name}: {errors}", end=" | ")
                else:
                    rospy.logwarn(f"Reference angles for {joint_name} not available.")

            print()
            self.rate.sleep()
        


if __name__ == "__main__":
    rospy.init_node("check_initial_pose", anonymous=True)

    parser = argparse.ArgumentParser(description="Check transforms or SUJ joint angles.")
    parser.add_argument("--type", choices=["transforms", "joint_angles"], required=True,
                        help="Type of checker to run: 'transforms' or 'SUJ_joint_angles'")
    args = parser.parse_args()

    # Example configuration dictionary
    config_dict = {
        "parent_frames": ["Cart",
                          "Cart",
                          "Cart",
                          "Cart",
                          "Cart",
                          "Cart"],

        "child_frames": ["PSM1_base",
                         "PSM2_base",
                         "ECM_base",
                         "PSM1",
                         "PSM2",
                         "ECM"],

        "ref_child_frames": ["PSM1_base_ref",
                             "PSM2_base_ref",
                             "ECM_base_ref",
                             "PSM1_ref",
                             "PSM2_ref",
                             "ECM_ref"],

        "suj_joint_angles_topics": {
            "PSM1": "/SUJ/PSM1/measured_js",
            "PSM2": "/SUJ/PSM2/measured_js",
            "ECM": "/SUJ/ECM/measured_js",
            "PSM3": "/SUJ/PSM3/measured_js"
        },

        "suj_joint_angles_ref_topics": {
            "PSM1": "/SUJ/PSM1/measured_js_ref",
            "PSM2": "/SUJ/PSM2/measured_js_ref",
            "ECM": "/SUJ/ECM/measured_js_ref",
            "PSM3": "/SUJ/PSM3/measured_js_ref"
        },
        "arm_names": ["PSM1", "PSM2", "PSM3"],
        "ecm_name": "ECM",
        "rospy_freq": 100,
        "significant_digits": 4
    }

    # Initialize the TransformChecker object
    checker = TransformChecker(config_dict)

    # Check the transforms
    # Run the appropriate checker based on the argument
    if args.type == "transforms":
        checker.check_transforms()
    elif args.type == "joint_angles":
        checker.check_suj_joint_angles()