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
from sensor_msgs.msg import JointState

class TransformPublisher:
    def __init__(self, config_dict):
        """
        Initialize the StaticTransformPublisher object.

        :param config_dict: Dictionary containing configuration parameters.
        """
        self.json_file = config_dict['json_file']
        self.parent_frame = config_dict["parent_frame"]  # Parent frame for all transforms
        self.ros_freq = config_dict["ros_freq"]  # Frequency for publishing transforms and jaw angles
        self.suj_joint_angles_suffix = config_dict['suj_joint_angles_suffix']

        self.data = self.load_data()
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.jaw_angles_ref_publisher = rospy.Publisher("jaw_angles_ref", JointState, queue_size=10)
        self.published_jaw_angles = False

        # Create publishers for each arm's SUJ joint angles
        self.suj_joint_angles_publishers = {
            arm_name: rospy.Publisher(f"/SUJ/{arm_name}/{self.suj_joint_angles_suffix}", JointState, queue_size=10)
            for arm_name in self.data["suj_joint_angles"].keys()
        }      
        self.published_suj_joint_angles = False

    def load_data(self):
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
        for key, transform_data in self.data["transforms"].items():
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
        rospy.loginfo("Static transforms of the reference stored locations published")

    def publish_suj_joint_angles(self):
        """
        Publish SUJ joint angles for each arm to individual topics.
        """
        for arm_name, arm_suj_joint_values in self.data["suj_joint_angles"].items():
            suj_joint_angles_msg = JointState()
            suj_joint_angles_msg.header.stamp = rospy.Time.now()

            suj_joint_angles_msg.name = [
                f"SUJ_{arm_name}_J{idx}" for idx in range(len(arm_suj_joint_values))
            ]
            suj_joint_angles_msg.position = arm_suj_joint_values

            # Publish to the specific topic for this arm
            self.suj_joint_angles_publishers[arm_name].publish(suj_joint_angles_msg)

            rospy.loginfo_once(f"SUJ joint angles for {arm_name} published to /SUJ/{arm_name}/{self.suj_joint_angles_suffix}")

    
    def publish_jaw_angles(self):
        """
        Publish the jaw angles to the "jaw_angles_ref" topic.
        """
        jaw_angles_msg = JointState()
        jaw_angles_msg.header.stamp = rospy.Time.now()
        jaw_angles_msg.name = list(self.data["jaw_angles"].keys())
        jaw_angles_msg.position = list(self.data["jaw_angles"].values())

        self.jaw_angles_ref_publisher.publish(jaw_angles_msg)

        if not self.published_jaw_angles:
            rospy.loginfo("Jaw angles published to 'jaw_angles_ref'.")
            
        self.published_jaw_angles = True

    def run(self):
        """
        Continuously publish transforms and jaw angles.
        """
        rate = rospy.Rate(self.ros_freq)  # Publish at 1 Hz
        self.publish_static_transforms()

        while not rospy.is_shutdown():
            self.publish_jaw_angles()
            self.publish_suj_joint_angles()
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node("static_transform_publisher", anonymous=True)

    # Path to the JSON file
    config_dict = {
        'json_file': "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot/ROS Packages/data_collection/utils_config/initial_pose_with_suj.json",
        'parent_frame': 'Cart',  # Parent frame for all transforms
        'ros_freq': 10,  # Frequency for publishing transforms and jaw angles
        'suj_joint_angles_suffix': 'measured_js_ref'  # Suffix for SUJ joint angles
    }

    # Create StaticTransformPublisher object and publish static transforms
    transform_publisher = TransformPublisher(config_dict)
    # transform_publisher.publish_static_transforms()
    transform_publisher.run()
    # # Keep the node alive
    # rospy.spin()