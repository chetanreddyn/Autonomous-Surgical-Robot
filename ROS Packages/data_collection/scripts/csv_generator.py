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
import csv
import argparse
import sys
import crtk
import shutil




class MessageSynchronizer:
    def __init__(self, config_dict):
        rospy.init_node('csv_generator', anonymous=True)

        self.queue_size = 10
        self.slop = 0.1
        # self.time_prev = time.time()
        self.topics = config_dict["topics"]
        self.time_format = config_dict["time_format"]
        self.logging_folder = config_dict["logging_folder"]
        self.logging_description = config_dict["logging_description"]
        
        self.csv_file = None
        self.image_save_folder = None
        self.csv_save_folder = None

        self.csv_columns = self.generate_csv_columns()
        self.frame_number = 0
        self.bridge = CvBridge()

        self.prev_time = time.time() # Time Stamp is the ROS time stamp uses the same reference as the system time
        self.prev_time_milli = 0
        self.time_sec = 0
        print(self.csv_columns)

        self.create_subscribers()

        # Synchronize the topics

        self.create_logging_folders()
        self.initialise_csv()

        self.ats = ApproximateTimeSynchronizer(self.subscribers, queue_size=self.queue_size, slop=self.slop)
        self.ats.registerCallback(self.callback)

    def create_subscribers(self):
        '''
        Creates subscribers for each topic
        '''
        self.subscribers = []
        for topic, msg_type in self.topics:
            self.subscribers.append(Subscriber(topic, msg_type))

    def create_logging_folders(self):
        '''
        Creates the logging folders if they do not exist
        '''
       
        self.final_logging_folder = os.path.join(self.logging_folder, self.logging_description)
        self.image_save_folder = os.path.join(self.final_logging_folder, "images")
        self.csv_save_folder = os.path.join(self.final_logging_folder, "csv")
       
        if os.path.exists(self.final_logging_folder):
            overwrite = input("Logging file already exists. Do you want to overwrite it? (y/n): ")
            if overwrite.lower() == 'y':
                print("Deleting the folder now")
                shutil.rmtree(self.final_logging_folder)
                rospy.loginfo(f"Logging folder {self.final_logging_folder} deleted")
                rospy.sleep(2)
            else:
                rospy.loginfo("rerun the script with a different logging description")
                sys.exit(0)


        os.makedirs(self.image_save_folder, exist_ok=True)
        os.makedirs(self.csv_save_folder, exist_ok=True)
        self.csv_file = os.path.join(self.csv_save_folder, "data.csv")


    def initialise_csv(self):
        '''
        Initialises the CSV file
        '''
        with open(self.csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.csv_columns)

    def callback(self, *msgs):
        # Process synchronized messages here
        # rospy.loginfo("Synchronized messages received")
        # self.duration = time.time() - self.time_prev
        # self.time_prev = time.time()
        self.time_milli = self.time_sec*1000
        duration = self.time_milli - self.prev_time_milli
        rospy.loginfo("Time elapsed in milliseconds: {:.0f} | Duration: {:.0f}".format(self.time_milli,duration))
        self.prev_time_milli = self.time_sec*1000
        # for msg in msgs:
        #     rospy.loginfo("Message timestamp: %s", msg.header.stamp.nsecs)
        # Add your processing code here
        
        self.process_messages(msgs)

    def process_timestamp(self, time_stamp):
        '''
        Processes the time stamp of the message
        '''
        message_time = datetime.fromtimestamp(time_stamp.to_sec())
        epoch_time_formatted = message_time.strftime(self.time_format)
        return epoch_time_formatted
    
    def process_messages(self, msgs):
        '''
        Generates a row of the CSV file
        '''
        pass

        # Safety hold for now
        # if self.frame_number > 50:
        #     return 
        
        time_stamp = msgs[0].header.stamp
        self.time_sec = time_stamp.to_sec() - self.prev_time
        epoch_time_formatted = self.process_timestamp(time_stamp)

        # print(epoch_time_formatted)
        row = [epoch_time_formatted,self.time_sec,self.frame_number]

        for i,(topic_name,topic_type) in enumerate(self.topics):
            # print(self.process_timestamp(msgs[i].header.stamp))
            if "cp" in topic_name:
                row.extend(self.process_pose_msg(topic_name, msgs[i]))
                
            elif "js" in topic_name:
                if "jaw" in topic_name:
                    row.extend(self.process_jaw_msg(topic_name,msgs[i]))
                else:
                    row.extend(self.process_joint_states_msg(topic_name,msgs[i]))

            elif "image" in topic_name:
                row.extend(self.process_image_msg(topic_name,msgs[i]))

        # print(len(row))
        self.write_csv_row(row)
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
        if "left" in topic_name:
            camera_name = "camera_left"
        elif "right" in topic_name:
            camera_name = "camera_right"

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        timestamp = self.process_timestamp(msg.header.stamp)
        image_path = os.path.join(self.image_save_folder, f'{camera_name}_{timestamp}.png')
        cv_image_reshaped = cv2.resize(cv_image, None, fx=0.4,fy=0.4)
        cv2.imwrite(image_path, cv_image_reshaped)
        # print(cv_image.shape)
        # rospy.loginfo("Saved image to %s", image_path)
        return [image_path]


    def generate_csv_columns(self):
        '''
        Generates the columns of the CSV file
        ['Epoch Time', 'Time (Seconds)', 'Frame Number', 'PSM1_joint_1', 'PSM1_joint_2', 'PSM1_joint_3', 'PSM1_joint_4', 'PSM1_joint_5', 'PSM1_joint_6', 'PSM1_jaw', 
        'PSM1_orientation_matrix_[1,1]', 'PSM1_orientation_matrix_[1,2]', 'PSM1_orientation_matrix_[1,3]', 
        'PSM1_orientation_matrix_[2,1]', 'PSM1_orientation_matrix_[2,2]', 'PSM1_orientation_matrix_[2,3]', 
        'PSM1_orientation_matrix_[3,1]', 'PSM1_orientation_matrix_[3,2]', 'PSM1_orientation_matrix_[3,3]', 
        'PSM2_joint_1', 'PSM2_joint_2', 'PSM2_joint_3', 'PSM2_joint_4', 'PSM2_joint_5', 'PSM2_joint_6', 'PSM2_jaw', 
        'PSM2_orientation_matrix_[1,1]', 'PSM2_orientation_matrix_[1,2]', 'PSM2_orientation_matrix_[1,3]', 
        'PSM2_orientation_matrix_[2,1]', 'PSM2_orientation_matrix_[2,2]', 'PSM2_orientation_matrix_[2,3]', 
        'PSM2_orientation_matrix_[3,1]', 'PSM2_orientation_matrix_[3,2]', 'PSM2_orientation_matrix_[3,3]', 
        'camera_right_image_path', 'camera_left_image_path']
        '''
        columns = ["Epoch Time","Time (Seconds)","Frame Number"]

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

    def write_csv_row(self, row):
        '''
        Writes a row to the CSV file
        '''
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    # List of topics and their message types
    argv = crtk.ral.parse_argv(sys.argv[1:])  # Skip argv[0], script name

    parser = argparse.ArgumentParser()

    parser.add_argument('-d','--logging_description',type=str,required=True,
                        help='Description of the data collection')
    
    parser.add_argument('-n','--logging_folder',type=str,default="/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Initial Samples",
                        help='Logging Folder')
    
    args = parser.parse_args(argv)

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
        "logging_description":args.logging_description,
        "logging_folder":args.logging_folder
    }
    synchronizer = MessageSynchronizer(config_dict)
    synchronizer.run()