#!/usr/bin/env python

'''
Author: Chetan Reddy Narayanaswamy

This Script Listens to the transform from base to stylus and transforms it to a more intuitive form using a transformation matrix given below
'''
import rospy
import numpy as np
import tf
from geometry_msgs.msg import PoseStamped

class PoseTransformer:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('omni_pose_transform', anonymous=True)

        # Define the static transformation matrix sc_T_po
        self.sc_T_po = np.array([[-1, 0, 0, 0],
                                 [0, 0, -1, 0],
                                 [0, -1, 0, 0],
                                 [0, 0, 0, 1]]) # It is a transform from surgeon console or assistant perspective to phantom omni

        # Create a tf listener
        self.listener = tf.TransformListener()

        # Create a publisher to the /phantom/pose_surgeon_console topic
        self.publisher = rospy.Publisher('/phantom/pose_assistant_perspective', PoseStamped, queue_size=10)

        # Set the rate at which to check for the transform
        self.rate = rospy.Rate(10.0)  # 10 Hz

    def transform_to_matrix(self, trans, rot):
        # Create the homogeneous transformation matrix from translation and rotation
        matrix = tf.transformations.quaternion_matrix(rot)
        matrix[0:3, 3] = trans
        return matrix

    def matrix_to_pose(self, matrix):
        # Extract translation from the matrix
        t = matrix[0:3, 3]

        # Extract rotation (quaternion) from the matrix
        q = tf.transformations.quaternion_from_matrix(matrix)

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
    
    def run(self):
        while not rospy.is_shutdown():
            try:
                # Look up the transform from base to stylus
                (trans, rot) = self.listener.lookupTransform('base', 'stylus', rospy.Time(0))

                # Convert the transform to a 4x4 transformation matrix po_T_pen
                po_T_pen = self.transform_to_matrix(trans, rot)

                # Compute the resulting transformation matrix sc_T_pen
                sc_T_pen = np.dot(self.sc_T_po, po_T_pen)

                # Convert the resulting matrix back to position and orientation
                transformed_pose = self.matrix_to_pose(sc_T_pen)

                # Create a new PoseStamped message
                transformed_msg = PoseStamped()
                transformed_msg.header.stamp = rospy.Time.now()
                transformed_msg.header.frame_id = 'base'
                transformed_msg.pose = transformed_pose

                # Publish the transformed PoseStamped message
                self.publisher.publish(transformed_msg)

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

            self.rate.sleep()

if __name__ == '__main__':
    try:
        # Create an instance of the PoseTransformer class
        transformer = PoseTransformer()

        # Run the transformer
        transformer.run()
    except rospy.ROSInterruptException:
        pass