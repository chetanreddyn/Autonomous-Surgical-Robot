#!/usr/bin/env python

import rospy
import tf2_ros
import json
from geometry_msgs.msg import TransformStamped


class TransformSaver:
    def __init__(self, config_dict):
        """
        Initialize the TransformSaver object.

        :param parents: List of parent frames.
        :param children: List of child frames.
        :param output_file: Path to the JSON file where transforms will be saved.
        """
        
        self.parents = config_dict['parents']
        self.children = config_dict['children'] 
        self.output_file = config_dict['output_file']

        if len(self.parents) != len(self.children):
            raise ValueError("Parents and children lists must have the same length.")
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def get_transform(self, parent, child):
        """
        Get the transform between a parent and child frame.

        :param parent: Parent frame.
        :param child: Child frame.
        :return: Transform as a dictionary.
        """
        try:
            transform = self.tf_buffer.lookup_transform(parent, child, rospy.Time(0), rospy.Duration(1.0))
            return {
                "translation": {
                    "x": transform.transform.translation.x,
                    "y": transform.transform.translation.y,
                    "z": transform.transform.translation.z,
                },
                "rotation": {
                    "x": transform.transform.rotation.x,
                    "y": transform.transform.rotation.y,
                    "z": transform.transform.rotation.z,
                    "w": transform.transform.rotation.w,
                },
            }
        except:
            rospy.logerr(f"Transform lookup failed for {parent} to {child}: {e}")
            return None

    def save_transforms(self):
        """
        Save the transforms between all parent and child frames to a JSON file.
        """
        transforms = {}
        for parent, child in zip(self.parents, self.children):
            rospy.loginfo(f"Getting transform from {parent} to {child}")
            transform = self.get_transform(parent, child)
            if transform:
                transforms[f"{parent}_to_{child}"] = transform

        print(f"Transforms: {transforms}")
        # Save to JSON file
        with open(self.output_file, "w") as file:
            json.dump(transforms, file, indent=4)
        rospy.loginfo(f"Transforms saved to {self.output_file}")


if __name__ == "__main__":
    rospy.init_node("save_initial_pose", anonymous=True)

    # Define parent and child frames
    parents = ["Cart", "Cart", "Cart", "Cart","Cart","Cart"]
    children = ["PSM1", "PSM1_base", "PSM2", "PSM2_base","ECM","ECM_base"]

    # Output file path
    output_file = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot/ROS Packages/data_collection/utils_config/initial_pose.json"

    config_dict = {
        "parents": parents,
        "children": children,
        "output_file": output_file
    }

    # Create TransformSaver object and save transforms
    transform_saver = TransformSaver(config_dict)
    transform_saver.save_transforms()
