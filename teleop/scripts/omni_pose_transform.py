#!/usr/bin/env python

'''
Author: Chetan Reddy Narayanaswamy

This Script Listens to the transform from base to stylus and transforms it to a more intuitive form using a transformation matrix given below
'''
import rospy
import numpy as np
import tf
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros


class PoseTransformer:
    def __init__(self,config_dict):
        # Initialize the ROS node
        rospy.init_node('omni_pose_transform', anonymous=True)


        self.target_topic = config_dict["target_topic"]
        self.camera_theta = config_dict["camera_theta"]
        self.eye_theta = config_dict["eye_theta"]

        self.eye_T_po = np.array([[1,0,0,0],
                                  [0,-np.sin(self.eye_theta),np.cos(self.eye_theta),0],
                                  [0,-np.cos(self.eye_theta),-np.sin(self.eye_theta),0],
                                  [0,0,0,1]])
        
        self.ecm_T_camera = np.array([[1,0,0,0],
                                      [0,np.cos(self.camera_theta),np.sin(self.camera_theta),0], 
                                      [0,-np.sin(self.camera_theta),np.cos(self.camera_theta),0],
                                      [0,0,0,1]])
        
        self.tip_T_psm = np.array([[1,0,0,0],
                                   [0,0,1,0],
                                   [0,-1,0,0],
                                    [0,0,0,1]])
        
        # self.camera_T_po = self.camera_T_ecm@self.ecm_T_po

        # Create a tf listener
        self.listener = tf.TransformListener()
        

        # Create a publisher to the /phantom/pose_surgeon_console topic
        self.publisher = rospy.Publisher(self.target_topic, PoseStamped, queue_size=10)

        # Set the rate at which to check for the transform
        self.rate = rospy.Rate(10.0)  # 10 Hz
        self.publish_static_transform()


    def publish_static_transform(self):
        broadcaster = tf2_ros.StaticTransformBroadcaster()
        static_transform_stamped = TransformStamped()

        static_transform_stamped.header.stamp = rospy.Time.now()
        static_transform_stamped.header.frame_id = "ECM"
        static_transform_stamped.child_frame_id = "camera"

        static_transform_stamped.transform.translation.x = self.ecm_T_camera[0, 3]
        static_transform_stamped.transform.translation.y = self.ecm_T_camera[1, 3]
        static_transform_stamped.transform.translation.z = self.ecm_T_camera[2, 3]

        quat = tf.transformations.quaternion_from_matrix(self.ecm_T_camera)
        static_transform_stamped.transform.rotation.x = quat[0]
        static_transform_stamped.transform.rotation.y = quat[1]
        static_transform_stamped.transform.rotation.z = quat[2]
        static_transform_stamped.transform.rotation.w = quat[3]

        broadcaster.sendTransform(static_transform_stamped)

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

                # Convert the transform to a 4x4 transformation matrix po_T_stylus
                po_T_stylus = self.transform_to_matrix(trans, rot)

                # Compute the resulting transformation matrix camera_T_stylus
                ecm_T_psm = self.ecm_T_camera@self.eye_T_po@po_T_stylus@self.tip_T_psm

                # Convert the resulting matrix back to position and orientation
                transformed_pose = self.matrix_to_pose(ecm_T_psm)

                # Create a new PoseStamped message
                transformed_msg = PoseStamped()
                transformed_msg.header.stamp = rospy.Time.now()
                transformed_msg.header.frame_id = 'base'
                transformed_msg.pose = transformed_pose

                rospy.loginfo("Published to :{}".format(self.target_topic))
                # Publish the transformed PoseStamped message
                self.publisher.publish(transformed_msg)

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

            self.rate.sleep()

if __name__ == '__main__':
    try:
        # Create an instance of the PoseTransformer class
        config_dict = {"target_topic":"/phantom/pose_assistant_perspective",
                       "camera_theta":40*np.pi/180,
                       "eye_theta":30*np.pi/180} # Zero means parallel to horizontal, 30 degrees means our line of sight is 30 degrees below the horizon

        transformer = PoseTransformer(config_dict)

        # Run the transformer
        transformer.run()
    except rospy.ROSInterruptException:
        pass