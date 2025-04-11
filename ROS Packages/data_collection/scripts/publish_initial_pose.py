#!/usr/bin/env python

import rospy
import tf2_ros
import json
from geometry_msgs.msg import TransformStamped


class TransformPublisher:
    def __init__(self, config_dict):
        """
        Initialize the TransformPublisher object.

        :param json_file: Path to the JSON file containing transforms.
        """
        self.json_file = config_dict['json_file']
        self.ros_freq = config_dict['ros_freq']
        self.parent_frame = config_dict["parent_frame"]  # Parent frame for all transforms

        self.transforms = self.load_transforms()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

    def load_transforms(self):
        """
        Load transforms from the JSON file.

        :return: Dictionary of transforms.
        """
        try:
            with open(self.json_file, "r") as file:
                return json.load(file)
        except Exception as e:
            rospy.logerr(f"Failed to load JSON file: {e}")
            return {}

    def publish_transforms(self):
        """
        Publish the transforms as TF frames.
        """
        rate = rospy.Rate(self.ros_freq)  # 10 Hz
        while not rospy.is_shutdown():
            for key, transform_data in self.transforms.items():
                parent_frame = self.parent_frame
                child_frame = key.split("_to_")[1] + "_ref"  # Append "_ref" to the child frame

                transform_msg = TransformStamped()
                transform_msg.header.stamp = rospy.Time.now()
                transform_msg.header.frame_id = parent_frame
                transform_msg.child_frame_id = child_frame

                # Set translation
                transform_msg.transform.translation.x = transform_data["translation"]["x"]
                transform_msg.transform.translation.y = transform_data["translation"]["y"]
                transform_msg.transform.translation.z = transform_data["translation"]["z"]

                # Set rotation
                transform_msg.transform.rotation.x = transform_data["rotation"]["x"]
                transform_msg.transform.rotation.y = transform_data["rotation"]["y"]
                transform_msg.transform.rotation.z = transform_data["rotation"]["z"]
                transform_msg.transform.rotation.w = transform_data["rotation"]["w"]

                # Publish the transform
                self.tf_broadcaster.sendTransform(transform_msg)

            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("transform_publisher", anonymous=True)


    # Path to the JSON file
    config_dict = {
        'json_file': "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot/ROS Packages/data_collection/utils_config/initial_pose.json",
        'ros_freq': 10,
        'parent_frame':'Cart'  # Frequency in Hz

    }

    # Create TransformPublisher object and publish transforms
    transform_publisher = TransformPublisher(config_dict)
    transform_publisher.publish_transforms()