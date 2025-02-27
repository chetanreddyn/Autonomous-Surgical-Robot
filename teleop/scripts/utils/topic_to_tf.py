#!/usr/bin/env python

import rospy
import tf2_ros
import argparse
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped


class PoseToTF2Broadcaster:
    def __init__(self, parent_frame, child_frame, topic_name):
        """
        Initializes the Pose to TF2 Broadcaster.
        """
        self.config_dict = {
            'parent_frame': parent_frame,
            'child_frame': child_frame,
            'topic_name': topic_name
        }

        # Initialize the ROS node
        rospy.init_node('pose_to_tf2_broadcaster')

        # Create a TransformBroadcaster
        self.br = tf2_ros.TransformBroadcaster()

        # Subscribe to the PoseStamped topic
        rospy.Subscriber(self.config_dict['topic_name'], PoseStamped, self.pose_callback)

        rospy.loginfo(f"Listening to {self.config_dict['topic_name']} and broadcasting TF from {self.config_dict['parent_frame']} to {self.config_dict['child_frame']}.")

    def pose_callback(self, msg):
        """
        Callback function for processing PoseStamped messages and broadcasting a TF2 transform.
        """
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = self.config_dict['parent_frame']
        transform.child_frame_id = self.config_dict['child_frame']

        # Set translation
        transform.transform.translation.x = msg.pose.position.x
        transform.transform.translation.y = msg.pose.position.y
        transform.transform.translation.z = msg.pose.position.z

        # Set rotation
        transform.transform.rotation = msg.pose.orientation

        # Broadcast the transform
        self.br.sendTransform(transform)

    def run(self):
        """
        Runs the ROS loop.
        """
        rospy.spin()

if __name__ == "__main__":
    # Use argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="Broadcast TF from a PoseStamped topic.")

    parser.add_argument("-parent_frame", type=str, help="Parent frame ID",default="base",required=False)
    parser.add_argument("-child_frame", type=str, help="Child frame ID",default="phantom_pose",required=False)
    parser.add_argument("-topic_name", type=str, help="PoseStamped topic name",default="/phantom/pose",required=False)

    args = parser.parse_args()

    # Create an instance of the broadcaster class
    broadcaster = PoseToTF2Broadcaster(args.parent_frame, args.child_frame, args.topic_name)

    # Run the broadcaster
    broadcaster.run()
