#!/usr/bin/env python
import rospy
import tf2_ros
import tf.transformations as tft
import argparse
import numpy as np

class TransformMatrixListener(object):
    """
    A class that listens to specified transforms and displays their
    multiplied 4x4 homogeneous transformation matrix.
    """
    def __init__(self, parent1_frame, child1_frame, parent2_frame, child2_frame):
        """
        Initializes the listener with the given parent and child frames.
        
        :param parent1_frame: The first parent frame ID.
        :param child1_frame: The first child frame ID.
        :param parent2_frame: The second parent frame ID.
        :param child2_frame: The second child frame ID.
        """
        self.parent1_frame = parent1_frame
        self.child1_frame = child1_frame
        self.parent2_frame = parent2_frame
        self.child2_frame = child2_frame
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.rate = rospy.Rate(10.0)  # 10 Hz update rate

    def lookup_transform(self, parent_frame, child_frame):
        """
        Attempts to lookup the transform between the parent and child frames.
        
        :return: A geometry_msgs/TransformStamped object.
        """
        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                parent_frame,
                child_frame,
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            return transform_stamped
        except Exception as e:
            rospy.logwarn("Failed to lookup transform from '%s' to '%s': %s", parent_frame, child_frame, e)
            return None

    def get_transformation_matrix(self, transform_stamped):
        """
        Converts the transform into a 4x4 matrix.
        
        :param transform_stamped: The transform from the tf buffer.
        :return: A 4x4 homogeneous transformation matrix.
        """
        # Extract translation and rotation (quaternion)
        translation = transform_stamped.transform.translation
        quaternion = transform_stamped.transform.rotation

        t = [translation.x, translation.y, translation.z]
        q = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]

        # Create the homogeneous transformation matrix
        matrix = tft.quaternion_matrix(q)
        matrix[0:3, 3] = t

        return matrix

    def display_transformation_matrix(self, matrix):
        """
        Displays the transformation matrix.
        
        :param matrix: The 4x4 homogeneous transformation matrix.
        """
        matrix_str = np.array2string(matrix, precision=2, separator=', ')
        cart_to_world_quaternion = tft.quaternion_from_matrix(matrix)
        cart_to_world_translation = matrix[:3,3]
        print(np.round(cart_to_world_translation,4),np.round(cart_to_world_quaternion,4))

        # rospy.loginfo("Final transformation matrix:\n%s", matrix_str)

    def run(self):
        """
        Main loop that listens for the transforms, multiplies them, and prints the final transformation matrix.
        """
        rospy.loginfo("Listening for transforms from '%s' to '%s' and from '%s' to '%s'...",
                      self.parent1_frame, self.child1_frame, self.parent2_frame, self.child2_frame)
        while not rospy.is_shutdown():
            transform1 = self.lookup_transform(self.parent1_frame, self.child1_frame)
            transform2 = self.lookup_transform(self.parent2_frame, self.child2_frame)
            if transform1 and transform2:
                matrix1 = self.get_transformation_matrix(transform1)
                matrix2 = self.get_transformation_matrix(transform2)
                final_matrix = matrix1@matrix2
                self.display_transformation_matrix(final_matrix)
            self.rate.sleep()

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Listen to transforms and display the multiplied transformation matrix."
    )
    parser.add_argument('-p1', '--parent1', required=True, help="First parent frame ID")
    parser.add_argument('-c1', '--child1', required=True, help="First child frame ID")
    parser.add_argument('-p2', '--parent2', required=True, help="Second parent frame ID")
    parser.add_argument('-c2', '--child2', required=True, help="Second child frame ID")
    
    # rospy.myargv() strips out ROS remapping arguments
    args = parser.parse_args(rospy.myargv()[1:])
    
    # Initialize the ROS node
    rospy.init_node('tf_matrix_listener', anonymous=True)
    
    # Create an instance of our TransformMatrixListener and run it
    listener = TransformMatrixListener(args.parent1, args.child1, args.parent2, args.child2)
    listener.run()