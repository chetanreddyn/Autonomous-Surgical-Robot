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

        # Initialize dVRK arms
        self.ecm_name = config_dict['arm_names'][0]
        self.arm1_name = config_dict['arm_names'][1]
        self.arm2_name = config_dict['arm_names'][2]
        self.num_transforms = len(config_dict['parent_frames'])
        self.parent_frames = config_dict['parent_frames']
        self.child_frames = config_dict['child_frames']
        self.arm_names = config_dict['arm_names']

        self.transform_lookup_wait_time = config_dict['transform_lookup_wait_time']
        self.arm1 = dvrk.psm(ral, self.arm1_name)
        self.arm2 = dvrk.psm(ral, self.arm2_name)
        self.ecm = dvrk.ecm(ral, self.ecm_name) # Arm 3 is ECM

        self.arm_objs = {self.arm1_name: self.arm1, self.arm2_name: self.arm2, self.ecm_name: self.ecm}
        self.move_cp_goals_received = False
        self.move_cp_goals = None # PyKDL.Frame() type transforms to send to move_cp topic

        # Initialize tf2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.loginfo("Loading transforms...")
        self.load_transforms()
        rospy.loginfo("move_cp goal positions for {} loaded successfully.".format(self.arm_names))
        # print(self.move_cp_goals)


    def load_transforms(self):
        
        self.move_cp_goals = {}

        for i in range(self.num_transforms):
            parent_frame = self.parent_frames[i]
            child_frame = self.child_frames[i]

            goal = None
            t0 = rospy.Time.now().to_sec()
            # print(t0)
            while not goal:
                goal = self.get_transform(parent_frame, child_frame)
                rospy.sleep(0.1)

                if rospy.Time.now().to_sec() - t0 > self.transform_lookup_wait_time:
                    rospy.logerr("Transform lookup timed out after {} seconds Looking for {} to {} Transform".format(
                        self.transform_lookup_wait_time,parent_frame, child_frame))
                    break
                
            self.move_cp_goals[parent_frame+"_to_"+child_frame] = goal


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

    def publish_transform(self, parent_frame, child_frame, arm_name):
        """
        Publish the transform from parent_frame to child_frame using move_cp.

        :param parent_frame: Parent frame name.
        :param child_frame: Child frame name.
        :param arm: dVRK arm object to move.
        """
        arm_obj = self.arm_objs[arm_name]
        goal = self.move_cp_goals.get(parent_frame+"_to_"+child_frame)
        if goal is None:
            rospy.logerr("Transform not found for {} to {}. Cannot publish.".format(parent_frame, child_frame))
            return
        
        else:
            rospy.loginfo("Moving {}".format(arm_name))
            arm_obj.move_cp(goal).wait(True)
            rospy.loginfo("Successfully moved {}".format(arm_name))


    def run(self):
        """
        Run the experiment initialization process.
        """
        rospy.loginfo("Initializing experiment...")
        rospy.sleep(1.0)
        # Publish transform from ECM_ref to PSM1_ref as ECM to PSM1


        for i in range(self.num_transforms):
            parent_frame = self.parent_frames[i]
            child_frame = self.child_frames[i]
            arm_name = self.arm_names[i]
            self.publish_transform(parent_frame,child_frame,arm_name)
            rospy.sleep(1)


        rospy.loginfo("Experiment initialization complete.")


if __name__ == "__main__":
    rospy.init_node("experiment_initializer", anonymous=True)

    # Configuration dictionary
    config_dict = {"parent_frames": ["Cart", "ECM_ref", "ECM_ref"],
                   "child_frames": ["ECM_ref", "PSM1_ref", "PSM2_ref"],
                   "arm_names": ["ECM", "PSM1", "PSM2"],
                   "transform_lookup_wait_time": 1.0
    }
    ral = crtk.ral('experiment_initializer')
    # Create ExperimentInitializer object and run the initialization
    initializer = ExperimentInitializer(ral,config_dict)
    initializer.run()