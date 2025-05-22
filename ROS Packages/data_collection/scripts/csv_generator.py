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
import json
import numpy as np


class MessageSynchronizer:
    def __init__(self, config_dict):

        self.queue_size = 10
        self.slop = 0.3
        # self.time_prev = time.time()
        self.topics = config_dict["topics"]
        self.time_format = config_dict["time_format"]
        self.logging_folder = config_dict["logging_folder"]
        self.logging_description = config_dict["logging_description"]
        self.duration = config_dict["duration"]
        self.total_num_steps = config_dict["total_num_steps"]
        self.image_size = config_dict["image_size"]
        self.loginfo = config_dict["loginfo"]
        self.rollout = config_dict["rollout"]
        self.dont_record_images = config_dict["dont_record_images"]
        self.exp_done = False

        self.arm_names = config_dict["arm_names"]
        
        self.csv_file = None
        self.image_save_folder = None
        self.csv_save_folder = None

        self.csv_columns = self.generate_csv_columns()
        self.frame_number = 0
        self.bridge = CvBridge()

        self.prev_time = time.time() # Time Stamp is the ROS time stamp uses the same reference as the system time
        self.prev_time_milli = 0
        self.time_sec = 0

        self.create_subscribers()

        # Synchronize the topics
        if self.rollout:
            rollout_started = False
            while not rollout_started and not rospy.is_shutdown():
                rollout_started = rospy.get_param("rollout_started", False) # Default value is False
                rospy.loginfo("Waiting for rollout node to start...")

                if not rollout_started:
                    rospy.sleep(0.1)

        self.create_logging_folders()
        self.initialise_csv()
        rospy.set_param("recording_started", True) # Reset the parameter to False
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
        # self.csv_save_folder = os.path.join(self.final_logging_folder, "csv")
       
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
        self.csv_file = os.path.join(self.final_logging_folder, "data.csv")
        self.meta_file = os.path.join(self.final_logging_folder, "experiment_config.json")


    def initialise_csv(self):
        '''
        Initialises the CSV file image_size   '''
        with open(self.csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.csv_columns)

    def callback(self, *msgs):
        # Process synchronized messages here
        # rospy.loginfo("Synchronized messages received")
        # self.duration = time.time() - self.time_prev
        # self.time_prev = time.time()

        if self.rollout:
            if self.frame_number>self.total_num_steps:
                if not self.exp_done:
                    rospy.loginfo("Rollout Duration Completed. Stopping the logging, restart the launch file to start a new experiment")
                    self.exp_done = True
                # rospy.signal_shutdown("Duration exceeded")
                return
            
        else:  # If not in rollout mode, check the duration 
            if self.time_sec>self.duration:
                if not self.exp_done:
                    rospy.loginfo("Experiment Duration Completed. Stopping the logging, restart the launch file to start a new experiment")
                    self.exp_done = True
                # rospy.signal_shutdown("Duration exceeded")
                return
        
        self.time_milli = self.time_sec*1000
        duration = self.time_milli - self.prev_time_milli
        if self.loginfo:
            rospy.loginfo("RECORDING DATA: Time elapsed (s): {:.3f} /{:2.0f} | Duration: {:.0f}".format(self.time_milli/1000, self.duration, duration))

        self.prev_time_milli = self.time_sec*1000

                    # for msg in msgs:
        #     rospy.loginfo("Message timestamp: %s", msg.header.stamp.nsecs)
        # Add your processing code here
        
        # print(self.csv_columns)
        self.process_messages(msgs)

    def process_timestamp(self, time_stamp):
        '''
        Processes the time stamp of the message
        '''
        message_time = datetime.fromtimestamp(time_stamp.to_sec())
        epoch_time_formatted = message_time.strftime(self.time_format)
        return epoch_time_formatted
    
    def check_pose_values(self, pose_values):
        '''
        Checks if the pose values are valid
        '''
        orientation_matrix = pose_values[:9]
        orientation_matrix = np.array(orientation_matrix).reshape(3, 3)

        orthogonality_check = np.allclose(np.dot(orientation_matrix, orientation_matrix.T), np.eye(3), atol=1e-6)

        # Check if the determinant is 1
        determinant_check = np.isclose(np.linalg.det(orientation_matrix), 1.0, atol=1e-6)

        return orthogonality_check and determinant_check

    def process_messages(self, msgs):
        '''
        Generates a row of the CSV file
        '''
        # rospy.loginfo(len(msgs))
        time_stamp = msgs[0].header.stamp
        self.time_sec = time_stamp.to_sec() - self.prev_time
        epoch_time_formatted = self.process_timestamp(time_stamp)

        # print(epoch_time_formatted)
        row = [epoch_time_formatted,self.time_sec,self.frame_number]

        for i,(topic_name,topic_type) in enumerate(self.topics):
            # print(self.process_timestamp(msgs[i].header.stamp))
            if "cp" in topic_name:
                pose_values = self.process_pose_msg(topic_name, msgs[i])
                
                if len(pose_values) != 12:
                    rospy.logwarn(f"Pose values are not a list of 12 elements, NOT LOGGING | Topic:{topic_name}")
                    return
                
                if not self.check_pose_values(pose_values):
                    rospy.logfatal(f"Orientation Matrix is not valid, NOT LOGGING | Topic:{topic_name}")
                    return
                
                row.extend(pose_values)
                
            elif "js" in topic_name:
                if "jaw" in topic_name:
                    jaw_value = self.process_jaw_msg(topic_name,msgs[i])
                    if len(jaw_value) != 1:
                        rospy.logwarn(f"Jaw value is not a singleton list, NOT LOGGING | Topic:{topic_name}")
                        return
                    row.extend(jaw_value)
                else:
                    joint_values = self.process_joint_states_msg(topic_name,msgs[i])

                    if len(joint_values) != 6:
                        rospy.logwarn(f"Joint values are not a list of 6 elements, NOT LOGGING | Topic:{topic_name}")
                        return
                    
                    row.extend(joint_values)

            elif "image" in topic_name:
                if self.dont_record_images:
                    continue
                else:
                    image_path = self.process_image_msg(topic_name,msgs[i])
                    if len(image_path) != 1:
                        rospy.logwarn(f"Image path is not a singleton set, NOT LOGGING | Topic:{topic_name}")
                        return
                    
                    row.extend(image_path)

        # print(len(row))rollout_lens):
        if len(row) != len(self.csv_columns):
            rospy.logwarn("Row length does not match the number of columns, NOT LOGGING")
            return
        else:      
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
        Processes messages of type JointState for the jaw columns
        '''
        jaw_angle_singleton_list = msg.position
        return jaw_angle_singleton_list
    
    def process_pose_msg(self, topic_name, msg):
        '''
        Processes messages of type PoseStamped
        Returns a list of x,y,z,orientation_matrix 9 elements
        '''
        # rospy.loginfo("Processing PoseStamped message")
        position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

        quaternion = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        rotation_matrix = tf_trans.quaternion_matrix(quaternion)[:3, :3]  # Extract the 3x3 rotation matrix
        flattened_matrix = rotation_matrix.flatten().tolist()  # Flatten the matrix to a list
        # rospy.loginfo("Position: %s, Orientation Matrix: %s", position, flattened_matrix)

        pose_values = flattened_matrix + position # The order is very important do not change it
        return pose_values
    
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
        cv_image_reshaped = cv2.resize(cv_image, None, fx=0.3,fy=0.3)
        assert cv_image_reshaped.shape[:2] == self.image_size
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

        for arm_name in self.arm_names:
            for k in range(1,4):
                for l in range(1,4):
                    columns.append(f"{arm_name}_orientation_matrix_[{k},{l}]")

            for direction in ["x", "y", "z"]:
                columns.append(f"{arm_name}_ee_{direction}")

            for i in range(1,7):
                columns.append(f"{arm_name}_joint_{i}")
            columns.append(f"{arm_name}_jaw")

        if not self.dont_record_images:
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

    def save_dict(self, data_dict):
        with open(self.meta_file, "w") as file:
            json.dump(data_dict, file, indent=4)

    def run(self):
        rospy.spin()

# class MetaFileGenerator:
#     def __init__(self,csv_generator_config_dict,meta_file_generator_config_dict):
#         self.meta_file_generator_config_dict = meta_file_generator_config_dict
#         self.csv_generator_config_dict = csv_generator_config_dict

#     def initialise_meta_file_dict(self):
#         meta_file_dict = {}
#         meta_file_dict["logging_description"] = self.csv_generator_config_dict["logging_description"]
#         meta_file_dict["logging_folder"] = self.csv_generator_config_dict["logging_folder"]
#         meta_file_dict["arm_names"] = self.csv_generator_config_dict["arm_names"]
#         meta_file_dict["teleop1_connection"] = self.meta_file_generator_config_dict["teleop1_connection"]
#         meta_file_dict["teleop2_connection"] = self.meta_file_generator_config_dict["teleop2_connection"]
#         meta_file_dict["teleop3_connection"] = self.meta_file_generator_config_dict["teleop3_connection"]
#         meta_file_dict["surgeon_name"] =  self.meta_file_generator_config_dict["surgeon_name"]
#         meta_file_dict["assistant_name"] = self.meta_file_generator_config_dict["assistant_name"]
#         meta_file_dict["tools_used"] = self.meta_file_generator_config_dict["tools_used"]
#         meta_file_dict["mtm_scale"] = self.meta_file_generator_config_dict["mtm_scale"]
#         meta_file_dict["phantom_omni_scale"] = self.meta_file_generator_config_dict["phantom_omni_scale"]
#         meta_file_dict["initial_pose_json_path"] = self.meta_file_generator_config_dict["initial_pose_json_path"]
#         meta_file_dict["Brightness"] = self.meta_file_generator_config_dict["Brightness"]

#     def get_tool_types(self):
#         pass


if __name__ == '__main__':
    rospy.init_node('csv_generator', anonymous=True)

    # LOGGING_FOLDER = rospy.get_param("LOGGING_FOLDER")
    LOGGING_FOLDER = rospy.get_param("LOGGING_FOLDER", "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Collaborative Expert Two Handed Object Transfer")
    # List of topics and their message types
    argv = crtk.ral.parse_argv(sys.argv[1:])  # Skip argv[0], script name

    parser = argparse.ArgumentParser()

    parser.add_argument('-d','--logging_description',type=str,required=True,
                        help='Description of the data collection')
    
    
    parser.add_argument('-T', '--duration', type=int, default=15,
                        help='Duration of the experiment in seconds')
    
    parser.add_argument('-N', '--total_num_steps',type=int,default=450)
    parser.add_argument('--dont_record_images', action="store_true", help="Disable image recording")
    
    parser.add_argument('--loginfo', action="store_true", help="Enable loginfo mode")
    parser.add_argument('--rollout', action="store_true", help="Enable rollout mode")
    # parser.add_argument('--logcamera', action="store_true", help="Enable logcamera mode")

    args, unknown = parser.parse_known_args()

    if LOGGING_FOLDER=="/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Initial Samples":
        rospy.logwarn("Logging Folder set to default 'Initial Samples'")
    

    topics = [
        ("/PSM1/setpoint_cp", PoseStamped),
        ("/PSM1/setpoint_js", JointState),
        ("/PSM1/jaw/setpoint_js", JointState),
    
        ("/PSM2/setpoint_cp", PoseStamped),
        ("/PSM2/setpoint_js", JointState),
        ("/PSM2/jaw/setpoint_js", JointState),

        ("/PSM3/setpoint_cp", PoseStamped),
        ("/PSM3/setpoint_js", JointState),
        ("/PSM3/jaw/setpoint_js", JointState)
    ]

    if not args.dont_record_images:
        topics.append(("/camera_right/image_raw", Image))
        topics.append(("/camera_left/image_raw", Image))
  
    csv_generator_config_dict = {
        "topics": topics,
        "time_format":"%Y-%m-%d %H:%M:%S.%f",
        "logging_description":args.logging_description,
        "logging_folder":LOGGING_FOLDER,
        "arm_names": ["PSM1", "PSM2", "PSM3"],
        "image_size": (324,576),
        "duration": args.duration,
        "total_num_steps": args.total_num_steps,
        "loginfo": args.loginfo,
        "rollout": args.rollout,
        "dont_record_images": args.dont_record_images
    }
    meta_file_dict = {}
    meta_file_dict["logging_description"] = csv_generator_config_dict["logging_description"]
    meta_file_dict["logging_folder"] = csv_generator_config_dict["logging_folder"]
    meta_file_dict["arm_names"] = csv_generator_config_dict["arm_names"]

    meta_file_dict["teleop1_connection"] = "MTMR-PSM1" # Always on the Console
    meta_file_dict["teleop2_connection"] = "Phantom-PSM2"
    meta_file_dict["teleop3_connection"] = None
    meta_file_dict["teleop1_name"] = "Alaa"
    meta_file_dict["teleop2_name"] = "Chetan"

    meta_file_dict["tools_used"] = ['FENESTRATED_BIPOLAR_FORCEPS:420205[..]','FENESTRATED_BIPOLAR_FORCEPS:420205[..]']
    meta_file_dict["mtm_scale"] = 0.4
    meta_file_dict["phantom_omni_scale"] = 0.4 
    meta_file_dict["initial_pose_json_path"] = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot/ROS Packages/data_collection/utils_config/initial_pose_with_suj.json"
    meta_file_dict["Brightness"] = 70
    meta_file_dict["Image Size"] = csv_generator_config_dict["image_size"]
    meta_file_dict["duration"] = csv_generator_config_dict["duration"]

    
    synchronizer = MessageSynchronizer(csv_generator_config_dict)
    synchronizer.save_dict(meta_file_dict)
    synchronizer.prev_time = time.time()
    synchronizer.run()