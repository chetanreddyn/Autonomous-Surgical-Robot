#!/usr/bin/env python

import rospy
import tf2_ros
import PyKDL
import dvrk
import crtk


class ExperimentInitializer:
    def __init__(self, ral, config_dict):
        """
        Initialize the ExperimentInitializer object.

        :param config_dict: Dictionary containing configuration parameters.
        """
        self.config_dict = config_dict

        # Initialize dVRK arms
        self.arm1_name = config_dict['arm1_name']
        self.arm2_name = config_dict['arm2_name']
        self.arm1 = dvrk.psm(ral, self.arm1_name)
        self.arm2 = dvrk.psm(ral, self.arm2_name)

        # Initialize tf2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def get_transform(self, parent_frame, child_frame):
        """
        Get the transform between a parent and child frame using tf2.

        :param parent_frame: Parent frame name.
        :param child_frame: Child frame name.
        :return: PyKDL.Frame object representing the transform.
        """
        try:
            transform = self.tf_buffer.lookup_transform(parent_frame, child_frame, rospy.Time(0), rospy.Duration(1.0))
            translation = transform.transform.translation
            rotation = transform.transform.rotation

            # Create a PyKDL.Frame object for the transform
            goal = PyKDL.Frame()
            goal.p = PyKDL.Vector(translation.x, translation.y, translation.z)
            goal.M = PyKDL.Rotation.Quaternion(rotation.x, rotation.y, rotation.z, rotation.w)

            return goal
        except Exception as e:
            rospy.logerr(f"Failed to get transform from {parent_frame} to {child_frame}: {e}")
            return None

    def publish_transform(self, parent_frame, child_frame, arm):
        """
        Publish the transform from parent_frame to child_frame using move_cp.

        :param parent_frame: Parent frame name.
        :param child_frame: Child frame name.
        :param arm: dVRK arm object to move.
        """
        goal = self.get_transform(parent_frame, child_frame)
        if goal:
            print("########GOAL Received",goal)
            arm.move_cp(goal).wait(True)

    def run(self):
        """
        Run the experiment initialization process.
        """
        rospy.loginfo("Initializing experiment...")
        rospy.sleep(1.0)
        # Publish transform from ECM_ref to PSM1_ref as ECM to PSM1
        self.publish_transform("ECM_ref", "{}_ref".format(self.arm1_name), self.arm1)
        rospy.sleep(1.0)

        # Publish transform from ECM_ref to PSM2_ref as ECM to PSM2
        self.publish_transform("ECM_ref", "{}_ref".format(self.arm2_name), self.arm2)

        rospy.loginfo("Experiment initialization complete.")


if __name__ == "__main__":
    rospy.init_node("experiment_initializer", anonymous=True)

    # Configuration dictionary
    config_dict = {"arm1_name": "PSM1",
                   "arm2_name": "PSM2"
    }
    ral = crtk.ral('experiment_initializer')
    # Create ExperimentInitializer object and run the initialization
    initializer = ExperimentInitializer(ral,config_dict)
    initializer.run()