#!/usr/bin/env python

import rospy
import numpy as np
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped

class PoseTransformer:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('omni_pose_transform', anonymous=True)

        # Define the static transformation matrix sc_T_po
        self.sc_T_po = np.array([[1, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, -1, 0, 0],
                                 [0, 0, 0, 1]])

        # Create a subscriber to the /phantom/omni topic
        self.subscriber = rospy.Subscriber('/phantom/pose', PoseStamped, self.callback)

        # Create a publisher to the /phantom/pose_surgeon_console topic
        self.publisher = rospy.Publisher('/phantom/pose_surgeon_console', PoseStamped, queue_size=10)

    def callback(self, msg):
        # print("Hereeee")
        # Convert PoseStamped to a 4x4 transformation matrix po_T_pen
        po_T_pen = self.pose_to_matrix(msg.pose)

        # Compute the resulting transformation matrix sc_T_pen
        sc_T_pen = np.dot(self.sc_T_po, po_T_pen)

        # Convert the resulting matrix back to position and orientation
        transformed_pose = self.matrix_to_pose(sc_T_pen)

        # Create a new PoseStamped message
        transformed_msg = PoseStamped()
        transformed_msg.header = msg.header
        transformed_msg.pose = transformed_pose

        # Publish the transformed PoseStamped message
        self.publisher.publish(transformed_msg)

    def pose_to_matrix(self, pose):
        # Extract translation and rotation (quaternion) from the pose
        t = [pose.position.x, pose.position.y, pose.position.z]
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

        # Create the homogeneous transformation matrix
        matrix = tft.quaternion_matrix(q)
        matrix[0:3, 3] = t

        return matrix

    def matrix_to_pose(self, matrix):
        # Extract translation from the matrix
        t = matrix[0:3, 3]

        # Extract rotation (quaternion) from the matrix
        q = tft.quaternion_from_matrix(matrix)

        # Create a Pose object
        pose = PoseStamped().pose
        pose.position.x = t[0]
        pose.position.y = t[1]
        pose.position.z = t[2]
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]

        return pose

if __name__ == '__main__':
    try:
        # Create an instance of the PoseTransformer class
        transformer = PoseTransformer()

        # Keep the node running
        rospy.spin()
    except rospy.ROSInterruptException:
        pass