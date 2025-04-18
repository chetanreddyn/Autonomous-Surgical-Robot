#!/usr/bin/env python

'''
check_initial_pose.py
Used to check if the Arm base has moved and how far it is from the saved reference frames
'''

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped


class TransformChecker:
    def __init__(self, config_dict):
        """
        Initialize the TransformChecker object.

        :param config_dict: Dictionary containing configuration parameters.
        """
        self.parent_frames = config_dict['parent_frames']
        self.child_frames = config_dict['child_frames']
        self.ref_child_frames = config_dict['ref_child_frames']
        self.rate = rospy.Rate(config_dict['rospy_freq'])
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

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
                if child_transform is None:
                    rospy.logwarn(f"Transform for {parent_frame} to {child_frame} not found.")
                    continue

                # Get the transform for the child_ref frame
                child_ref_transform = self.get_transform(parent_frame, ref_child_frame)
                if child_ref_transform is None:
                    rospy.logwarn(f"Transform for {parent_frame} to {ref_child_frame} not found.")
                    continue

                # Calculate the error metric
                translation_error, rotation_error = self.calculate_error(child_transform, child_ref_transform)
                print(f"{child_frame:<5}: ({translation_error:.3f},{rotation_error:.3f})",end=" | ")
            print()
            self.rate.sleep()



if __name__ == "__main__":
    rospy.init_node("check_initial_pose", anonymous=True)

    # Example configuration dictionary
    config_dict = {
        "parent_frames": ["Cart", "Cart", "Cart"],
        "child_frames": ["PSM1_base", "PSM2_base", "ECM_base"],
        "ref_child_frames": ["PSM1_base_ref", "PSM2_base_ref", "ECM_base_ref"],
        "rospy_freq": 100
    }

    # Initialize the TransformChecker object
    checker = TransformChecker(config_dict)

    # Check the transforms
    checker.check_transforms()