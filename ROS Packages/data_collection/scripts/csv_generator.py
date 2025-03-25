#!/usr/bin/env python

import rospy
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image,JointState
from geometry_msgs.msg import PoseStamped
import time
from datetime import datetime
import tf.transformations as tf_trans
import cv2
import os
from cv_bridge import CvBridge



class MessageSynchronizer:
    def __init__(self, config_dict):
        rospy.init_node('csv_generator', anonymous=True)

        self.queue_size = 10
        self.slop = 1
        self.time_prev = time.time()
        self.topics = config_dict["topics"]
        self.time_format = config_dict["time_format"]
        self.image_save_folder = config_dict["image_save_folder"]
        self.csv_columns = self.generate_csv_columns()
        self.frame_number = 0
        self.bridge = CvBridge()

        print(self.csv_columns)

        # Create subscribers for each topic
        self.subscribers = []
        for topic, msg_type in self.topics:
            self.subscribers.append(Subscriber(topic, msg_type))

        # Synchronize the topics
        self.ats = ApproximateTimeSynchronizer(self.subscribers, queue_size=self.queue_size, slop=self.slop)
        self.ats.registerCallback(self.callback)

    def callback(self, *msgs):
        # Process synchronized messages here
        rospy.loginfo("Synchronized messages received")
        # self.duration = time.time() - self.time_prev
        # self.time_prev = time.time()
        # rospy.loginfo("Time elapsed: {:.0f}".format(1/self.duration))
        # for msg in msgs:
        #     rospy.loginfo("Message timestamp: %s", msg.header.stamp.nsecs)
        # Add your processing code here
        
        self.process_messages(msgs)

    def process_timestamp(self, time_stamp):
        '''
        Processes the time stamp of the message
        '''
        message_time = datetime.fromtimestamp(time_stamp.to_sec())
        formatted_time = message_time.strftime(self.time_format)
        return formatted_time
    
    def process_messages(self, msgs):
        '''
        Generates a row of the CSV file
        '''
        pass

        # Safety hold for now
        if self.frame_number > 50:
            return 
        
        time_stamp = msgs[0].header.stamp
        formatted_time = self.process_timestamp(time_stamp)

        print(formatted_time)
        row = [formatted_time,self.frame_number]

        for i,(topic_name,topic_type) in enumerate(self.topics):
            if "cp" in topic_name:
                row.extend(self.process_pose_msg(topic_name, msgs[i]))
            elif "js" in topic_name:
                if "jaw" in topic_name:
                    row.extend(self.process_jaw_msg(topic_name,msgs[i]))
                else:
                    row.extend(self.process_joint_states_msg(topic_name,msgs[i]))

            elif "image" in topic_name:
                row.extend(self.process_image_msg(topic_name,msgs[i]))

        # print(row)
        self.frame_number += 1


    def process_joint_states_msg(self, topic_name, msg):
        '''
        Processes messages of type JointState 
        Returns a list of 6 joint angles
        '''
        return msg.position
    
    def process_jaw_msg(self, topic_name, msg):
        '''
        Processes messages of type JointState for the jaw
        Returns the jaw angle
        '''
        jaw_angle_singleton_list = msg.position
        return jaw_angle_singleton_list
    
    def process_pose_msg(self, topic_name, msg):
        '''
        Processes messages of type PoseStamped
        Returns a list of x,y,z,orientation_matrix 9 elements
        '''
        # rospy.loginfo("Processing PoseStamped message")
        quaternion = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        rotation_matrix = tf_trans.quaternion_matrix(quaternion)[:3, :3]  # Extract the 3x3 rotation matrix
        flattened_matrix = rotation_matrix.flatten().tolist()  # Flatten the matrix to a list
        # rospy.loginfo("Position: %s, Orientation Matrix: %s", position, flattened_matrix)
        return flattened_matrix
    
    def process_image_msg(self, topic_name, msg):
        '''
        Processes messages of type Image
        Saves the image and returns the path
        '''
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        timestamp = self.process_timestamp(msg.header.stamp)
        image_path = os.path.join(self.image_save_folder, f'image_{timestamp}.png')
        cv2.imwrite(image_path, cv_image)
        # rospy.loginfo("Saved image to %s", image_path)
        return [image_path]


    def generate_csv_columns(self):
        '''
        Generates the columns of the CSV file
        ['Time', 'Frame Number', 'PSM1_joint_1', 'PSM1_joint_2', 'PSM1_joint_3', 'PSM1_joint_4', 'PSM1_joint_5', 'PSM1_joint_6', 'PSM1_jaw', 
        'PSM1_orientation_matrix_[1,1]', 'PSM1_orientation_matrix_[1,2]', 'PSM1_orientation_matrix_[1,3]', 
        'PSM1_orientation_matrix_[2,1]', 'PSM1_orientation_matrix_[2,2]', 'PSM1_orientation_matrix_[2,3]', 
        'PSM1_orientation_matrix_[3,1]', 'PSM1_orientation_matrix_[3,2]', 'PSM1_orientation_matrix_[3,3]', 
        'PSM2_joint_1', 'PSM2_joint_2', 'PSM2_joint_3', 'PSM2_joint_4', 'PSM2_joint_5', 'PSM2_joint_6', 'PSM2_jaw', 
        'PSM2_orientation_matrix_[1,1]', 'PSM2_orientation_matrix_[1,2]', 'PSM2_orientation_matrix_[1,3]', 
        'PSM2_orientation_matrix_[2,1]', 'PSM2_orientation_matrix_[2,2]', 'PSM2_orientation_matrix_[2,3]', 
        'PSM2_orientation_matrix_[3,1]', 'PSM2_orientation_matrix_[3,2]', 'PSM2_orientation_matrix_[3,3]', 
        'camera_right_image_path', 'camera_left_image_path']
        '''
        columns = ["Time","Frame Number"]

        for arm_name in ["PSM1", "PSM2"]:
            for i in range(1,7):
                columns.append(f"{arm_name}_joint_{i}")
            columns.append(f"{arm_name}_jaw")

            for k in range(1,4):
                for l in range(1,4):
                    columns.append(f"{arm_name}_orientation_matrix_[{k},{l}]")

        for camera_name in ["camera_right", "camera_left"]:
            columns.append(f"{camera_name}_image_path")

        return columns

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    # List of topics and their message types
    topics = [
        ("/PSM1/setpoint_cp", PoseStamped),
        ("/PSM1/setpoint_js", JointState),
        ("/PSM1/jaw/setpoint_js", JointState),
    
        ("/PSM2/setpoint_cp", PoseStamped),
        ("/PSM2/setpoint_js", JointState),
        ("/PSM2/jaw/setpoint_js", JointState),

        ("/camera_right/image_raw", Image),
        ("/camera_left/image_raw", Image)
    ]

    
  
    config_dict = {
        "topics": topics,
        "time_format":"%Y-%m-%d %H:%M:%S.%f",
        "image_save_folder":"/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot/ROS Packages/data_collection/logs/images_tmp"
    }
    synchronizer = MessageSynchronizer(config_dict)
    synchronizer.run()