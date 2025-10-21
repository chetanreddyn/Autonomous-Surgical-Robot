#!/usr/bin/env python

import rospy
import tf2_ros
import json
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState



class PoseSaver:
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
        self.ecm_name = config_dict['ecm_name']
        self.arm_names = config_dict['arm_names']

        self.jaw_names = [arm_name+"_jaw" for arm_name in self.arm_names]
        self.jaw_topic_suffix = config_dict['jaw_topic_suffix']

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.transforms = None
        self.loaded_transforms = False

        self.jaw_angles = {jaw_name: None for jaw_name in self.jaw_names}
        self.loaded_jaw_angles = False

        self.suj_joint_angles_suffix = config_dict['suj_joint_angles_suffix']
        self.suj_joint_angles_dict = {
            arm_name: None
            for arm_name in (self.arm_names + [self.ecm_name])
            }
        
        self.loaded_suj_joint_angles = False
        


        # Subscribe to jaw topics
        for i, arm_name in enumerate(self.arm_names):

            topic = arm_name + self.jaw_topic_suffix
            rospy.Subscriber(topic, JointState, 
                             self.jaw_callback,
                             callback_args=self.jaw_names[i])
            
        for i, arm_name in enumerate(self.suj_joint_angles_dict):
            topic = "/SUJ/" + arm_name + self.suj_joint_angles_suffix
            rospy.Subscriber(topic, JointState,
                             self.suj_joint_angles_callback,
                             callback_args=arm_name)

        self.saved_to_json = False
            
    def suj_joint_angles_callback(self, msg, arm_name):
        """
        Callback function for SUJ Joint Angles topic subscription.

        param msg: JointState message.
        param arm_name: Name of the arm.
        """
        if self.loaded_suj_joint_angles:
            return
        
        # Extract joint angles from the message
        suj_joint_angles = msg.position

        self.suj_joint_angles_dict[arm_name] = suj_joint_angles

        for arm_name in self.suj_joint_angles_dict:
            if self.suj_joint_angles_dict[arm_name] is None:
                break
        else:
            self.loaded_suj_joint_angles = True
            rospy.loginfo("All SUJ joint angles loaded.")


    def jaw_callback(self, msg, jaw_name):
        """
        Callback function for jaw topic subscription.

        :param msg: JointState message.
        :param arm_name: Name of the arm.
        """
        if self.loaded_jaw_angles:
            return 
        
        # Extract jaw angles from the message
        jaw_angle = msg.position
        # rospy.loginfo(f"Jaw angles for {arm_name}: {jaw_angle}")
        self.jaw_angles[jaw_name] = jaw_angle[0]


        for jaw_name in self.jaw_angles:
            if self.jaw_angles[jaw_name] is None:
                break
        else:
            self.loaded_jaw_angles = True
            rospy.loginfo("All jaw angles loaded.")

        
         # Save jaw angles to a JSON file or process them as needed


                
        # Save jaw angles to a JSON file or process them as needed

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
        except Exception as e:
            rospy.logerr(f"Transform lookup failed for {parent} to {child}: {e}")
            return None

    def load_transforms(self):
        """
        Save the transforms between all parent and child frames to a JSON file.
        """
        self.transforms = {}
        for parent, child in zip(self.parents, self.children):
            rospy.loginfo(f"Getting transform from {parent} to {child}")
            transform = self.get_transform(parent, child)
            if transform:
                self.transforms[f"{parent}_to_{child}"] = transform


        if len(self.transforms) == len(self.parents):
            self.loaded_transforms = True
            rospy.loginfo("All transforms loaded.")

        # # Save to JSON file
        # with open(self.output_file, "w") as file:
        #     json.dump(transforms, file, indent=4)
        # rospy.loginfo(f"Transforms saved to {self.output_file}")
        # self.saved_transforms = True

    def save_to_json(self):
        """
        Save the Transforms and Jaw angles to a JSON file.
        """
        if not self.loaded_transforms or not self.loaded_jaw_angles:
            rospy.logwarn("Cannot save to JSON. Ensure both transforms and jaw angles are loaded.")
            return

        data = {
            "transforms": self.transforms,
            "jaw_angles": self.jaw_angles,
            "suj_joint_angles": self.suj_joint_angles_dict
        }

        with open(self.output_file, "w") as file:
            json.dump(data, file, indent=4)
        rospy.loginfo(f"Transforms and jaw angles saved to {self.output_file}")
        
    def run(self):
        """
        Main method to load transforms and wait for jaw angles before saving.
        """
        rospy.loginfo("Loading transforms...")
        self.load_transforms()

        rospy.loginfo("Waiting for jaw angles and suj_joint_angles...")
        while not rospy.is_shutdown() and not self.saved_to_json:
            if self.loaded_transforms and self.loaded_jaw_angles and self.loaded_suj_joint_angles:
                self.save_to_json()
                rospy.signal_shutdown("Data saved successfully.")
                self.saved_to_json = True
            rospy.sleep(1)

if __name__ == "__main__":
    rospy.init_node("save_initial_pose", anonymous=True)

    # Define parent and child frames
    parents = ["Cart", "Cart", "Cart", "Cart", "Cart", "Cart",
               "Cart", "Cart"]
    children = ["PSM1", "PSM1_base", "PSM2", "PSM2_base", "PSM3", "PSM3_base",
                "ECM", "ECM_base"]

    # Output file path
    output_file = (
        "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot/"
        "ROS Packages/data_collection/utils_config/NVIDIA_data_collection.json"
    ) 

    config_dict = {
        "parents": parents,
        "children": children,
        "output_file": output_file,
        "arm_names": ["PSM1", "PSM2", "PSM3"],
        "ecm_name": "ECM",
        "jaw_topic_suffix": "/jaw/setpoint_js",
        "suj_joint_angles_suffix": "/measured_js",
    }

    # Create TransformSaver object and save transforms
    pose_saver = PoseSaver(config_dict)
    pose_saver.run()
