#!/usr/bin/env python

'''
publish_initial_pose.py
Reads the JSON file with the saved locations of the arms and publishes the static transforms
so that the reference frames can be visualised in rviz
'''
import rospy
import tf2_ros
import json
from geometry_msgs.msg import TransformStamped


class TransformPublisher:
    def __init__(self, config_dict):
        """
        Initialize the StaticTransformPublisher object.

        :param config_dict: Dictionary containing configuration parameters.
        """
        self.json_file = config_dict['json_file']
        self.parent_frame = config_dict["parent_frame"]  # Parent frame for all transforms

        self.transforms = self.load_transforms()
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

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

    def publish_static_transforms(self):
        """
        Publish the static transforms as TF frames.
        """
        static_transforms = []
        for key, transform_data in self.transforms.items():
            parent_frame, child_frame = key.split("_to_")
            child_frame += "_ref"  # Append "_ref" to the child frame

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

            static_transforms.append(transform_msg)

        # Publish all static transforms at once
        self.tf_broadcaster.sendTransform(static_transforms)
        rospy.loginfo("Static transforms published.")


if __name__ == "__main__":
    rospy.init_node("static_transform_publisher", anonymous=True)

    # Path to the JSON file
    config_dict = {
        'json_file': "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot/ROS Packages/data_collection/utils_config/initial_pose_11_apr.json",
        'parent_frame': 'Cart'  # Parent frame for all transforms
    }

    # Create StaticTransformPublisher object and publish static transforms
    transform_publisher = TransformPublisher(config_dict)
    transform_publisher.publish_static_transforms()

    # Keep the node alive
    rospy.spin()