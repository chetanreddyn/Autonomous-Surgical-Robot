#!/usr/bin/env python
import rospy


from publish_initial_pose import TransformPublisher

def main():
    rospy.init_node("transform_publisher", anonymous=True)


    # Path to the JSON file
    config_dict_publish_ref = {
        'json_file': "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot/ROS Packages/data_collection/utils_config/initial_pose.json",
        'ros_freq': 10,
        'parent_frame':'Cart'  # Frequency in Hz

    }

    # Create TransformPublisher object and publish transforms
    transform_publisher = TransformPublisher(config_dict_publish_ref)
    transform_publisher.publish_static_transforms()

if __name__ == "__main__":
    main()