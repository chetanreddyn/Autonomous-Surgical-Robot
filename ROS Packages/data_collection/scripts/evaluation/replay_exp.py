#!/usr/bin/env python

import rospy
import crtk
import sys
import numpy as np
import dvrk
import csv

class ReplayExperiment:
    def __init__(self, ral, config_dict):
        """
        Initialize the ReplayExperiment object.

        :param csv_file: Path to the CSV file containing joint angles.
        :param arm_name: Name of the dVRK arm to control (default: PSM1).
        """
        self.csv_file = config_dict['csv_file']  # Path to the CSV file
        self.arm_name1 = config_dict['arm_name1']  # Name of the dVRK arm
        self.ros_freq = config_dict['ros_freq'] 
        self.joint_columns = config_dict['joint_columns']  # Joint columns (1 to 6)
        
        self.arm = dvrk.psm(ral, self.arm_name1)  # Initialize the dVRK arm
        # self.joint_columns = [f"{self.arm_name1}_joint_{i}" for i in range(1, 7)]  # Joint columns (1 to 6)
        rospy.loginfo(f"Initialized ReplayExperiment for {self.arm_name1}")

    def read_csv(self):
        """
        Read the CSV file and extract joint angles for the specified arm.

        :return: List of joint angle arrays.
        """
        joint_angles = []
        try:
            with open(self.csv_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Extract joint angles for the arm
                    angles = [float(row[column]) for column in self.joint_columns]
                    joint_angles.append(angles)
            rospy.loginfo(f"Successfully read {len(joint_angles)} joint angle sets from CSV.")
        except Exception as e:
            rospy.logerr(f"Failed to read CSV file: {e}")
        return joint_angles

    def move_arm(self, joint_angles):
        """
        Move the arm to the specified joint angles.

        :param joint_angles: List of joint angle arrays.
        """
        rospy.loginfo("Starting arm movement...")
        rate = rospy.Rate(self.ros_freq)  # 10 Hz
        for angles in joint_angles:
            if not rospy.is_shutdown():
                # rospy.loginfo(f"Moving to joint angles: {angles}")
                self.arm.servo_jp(np.array(angles[:-1]))  # Move to the joint position
                self.arm.jaw.servo_jp(np.array([angles[-1]]))  # Move the jaw to the specified angle
                print(np.array(angles))

                rate.sleep()
            
            else:
                rospy.loginfo("Shutting Down")
                break
        else:
            rospy.loginfo("Finished arm movement.")

    def run(self):
        """
        Run the replay experiment.
        """
        joint_angles = self.read_csv()
        print((joint_angles[0]))
        
        if joint_angles:
            self.move_arm(joint_angles)


if __name__ == "__main__":
    argv = crtk.ral.parse_argv(sys.argv[1:])  # Skip argv[0], script name

    # Path to the CSV file
    csv_file = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Object-Transfer/Demo1/data.csv"

    config_dict = {"csv_file": csv_file,
                   "arm_name1": "PSM1",
                   "ros_freq": 30,
                   "joint_columns": ["PSM1_orientation_matrix_[1,3]",
                                     "PSM1_orientation_matrix_[2,1]",
                                     "PSM1_orientation_matrix_[2,2]",
                                     "PSM1_orientation_matrix_[2,3]",
                                     "PSM1_orientation_matrix_[3,1]",
                                     "PSM1_orientation_matrix_[3,2]",
                                     "PSM1_orientation_matrix_[3,3]",]
                    }  # Specify the arm name if needed
    # Create ReplayExperiment object and execute the replay
    
    ral = crtk.ral('mimic_pose')

    replay_exp = ReplayExperiment(ral, config_dict)
    replay_exp.run()
    # ral.spin_and_execute(replay_exp.run)