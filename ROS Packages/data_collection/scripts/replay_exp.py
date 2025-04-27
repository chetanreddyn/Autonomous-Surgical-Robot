#!/usr/bin/env python

import rospy
import crtk
import sys
import numpy as np
import dvrk
import csv
from typing import Dict, List
import argparse

from sensor_msgs.msg import JointState

from initialize_exp import ExperimentInitializer


class ReplayExperiment:
    def __init__(self, ral, config_dict):
        """
        Initialize the ReplayExperiment object.

        :param csv_file: Path to the CSV file containing joint angles.
        :param arm_name: Name of the dVRK arm to control (default: PSM1).
        """
        self.csv_file = config_dict['csv_file']  # Path to the CSV file
        self.arm_name1 = config_dict['arm_names'][0]  # Name of the dVRK arm
        self.arm_name2 = config_dict['arm_names'][1]  # Name of the dVRK arm
        self.arm_names = config_dict['arm_names']  # List of arm names
        self.ros_freq = config_dict['ros_freq'] 
        # self.joint_columns_arm1 = config_dict['joint_columns_arm1']  # Joint columns (1 to 7)
        # self.joint_columns_arm2 = config_dict['joint_columns_arm2']
        
        self.joint_columns = {self.arm_name1: self.generate_joint_columns(self.arm_name1),
                              self.arm_name2: self.generate_joint_columns(self.arm_name2)}  # Joint columns (1 to 7)
        
        self.initial_joint_states = {self.arm_name1: None, self.arm_name2: None}  # Initial joint states for each arm
        self.initial_joint_states_updated = False
        self.initial_joint_state_dicrepancy_tolerance = config_dict['initial_joint_state_dicrepancy_tolerance']
        
        self.arm_objs = config_dict['arm_objs']  # Dictionary of arm objects
        
        # self.joint_columns = [f"{self.arm_name1}_joint_{i}" for i in range(1, 7)]  # Joint columns (1 to 6)
        rospy.loginfo(f"Initialized ReplayExperiment for {self.arm_name1} and {self.arm_name2} with CSV file: {self.csv_file}")


    def get_joint_angles(self, row, arm_name):
        joint_angles = [float(row[column]) for column in self.joint_columns[arm_name]]
        return joint_angles

    def read_csv(self):
        """
        Read the CSV file and extract joint angles for the specified arm.

        :return: List of joint angle arrays.
        """
        joint_angles_trajectories = {self.arm_name1: [], self.arm_name2: []}
        try:
            with open(self.csv_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader: # Each row is a dictionary with column names as keys
                    # Extract joint angles for the arm

                    # Extracting for each arm
                    for arm_name in joint_angles_trajectories.keys():
                        joint_angles_trajectories[arm_name].append(self.get_joint_angles(row, arm_name))
            
            # Convert the list of joint angles to a numpy array
            rospy.loginfo(f"Successfully read joint angle sets from CSV.")
        except Exception as e:
            rospy.logerr(f"Failed to read CSV file: {e}")
            return None

        return joint_angles_trajectories  # Return the dictionary of joint angles

        
    def move_arms(self, joint_angles_trajectories: Dict[str, List[List[float]]]):
        """
        Move the arm to the specified joint angles.

        joint_angles_trajectories: Dictionary of Lists
        """
        rospy.loginfo("Starting arm movement...")
        rate = rospy.Rate(self.ros_freq)  

        trajectory_length = len(joint_angles_trajectories[self.arm_name1])
        t0 = rospy.Time.now()
        for t in range(trajectory_length):
            elapsed_time = (rospy.Time.now() - t0).to_sec()
            if not rospy.is_shutdown():
                rospy.loginfo(f"Elapsed time: {elapsed_time:.2f} s | Step: {t}/{trajectory_length}")
    
                # Move each arm to the specified joint angles
                for arm_name in joint_angles_trajectories.keys():
                    angles = joint_angles_trajectories[arm_name][t]
                    # rospy.loginfo(f"Moving {arm_name} to joint angles: {angles}")

                    initial_joint_state = np.array(self.initial_joint_states[arm_name])
                    target_joint_state = np.array(angles[:-1])
                    # print(initial_joint_state.round(2),target_joint_state.round(2))
                    diff = np.abs(initial_joint_state - target_joint_state).max()
                    
                    if diff > 2*self.initial_joint_state_dicrepancy_tolerance:
                        rospy.logfatal(f"Joint state discrepancy for {arm_name} is too large | max joint discrepancy{diff:.2f}")
                        return 
                
                    elif diff > self.initial_joint_state_dicrepancy_tolerance:
                        rospy.logwarn(f"Joint state discrepancy for {arm_name} exceeds tolerance | max joint discrepancy {diff:.2f}")

                    else:
                        self.arm_objs[arm_name].move_jp(target_joint_state)
                        self.arm_objs[arm_name].jaw.move_jp(np.array([angles[-1]])) 

                rate.sleep()

            else:
                rospy.loginfo("Shutting Down")
                break
                    
        else:
            rospy.loginfo("Finished arm movement.")

    def generate_joint_columns(self, arm_name):
        joint_columns_arm = []
        for i in range(1, 7):
            joint_columns_arm.append(f"{arm_name}_joint_{i}")

        joint_columns_arm.append(f"{arm_name}_jaw")
        return joint_columns_arm

    def update_initial_joint_state(self, msg, arm_name):
        self.initial_joint_states[arm_name] = msg.position

        for arm_name in self.arm_names:
            if self.initial_joint_states[arm_name] is None:
                break
        else:
            self.initial_joint_states_updated = True
            # rospy.loginfo(f"Initial joint states updated for {arm_name}.")

    def run(self):
        """
        Run the replay experiment.
        """
        joint_angles_trajectories = self.read_csv()

        for arm_name in self.arm_names:
            rospy.Subscriber(f'/{arm_name}/setpoint_js',JointState, self.update_initial_joint_state, callback_args=arm_name)

        while not self.initial_joint_states_updated and not rospy.is_shutdown():
            rospy.loginfo("Waiting for initial joint states to be updated...")
            rospy.sleep(0.1)
        
        rospy.loginfo("Initial joint states updated for both arms.")
        if joint_angles_trajectories and self.initial_joint_states_updated:
            self.move_arms(joint_angles_trajectories)
            
        else:
            joint_angles_trajectories_loaded = joint_angles_trajectories is not None
            rospy.logerr("Cannot proceed with arm movement | joint_angles_trajectories_loaded: {joint_angles_trajectories_loaded} | initial_joint_states_updated: {self.initial_joint_states_updated}")
            return


if __name__ == "__main__":
    argv = crtk.ral.parse_argv(sys.argv[1:])  # Skip argv[0], script name
    parser = argparse.ArgumentParser(description="Replay Experiment")

    parser.add_argument('-d', '--demo_number', type=str, required=True, help="Demo number to replay")
    parser.add_argument('-r', '--reposition_ecm', action='store_true', help="Reposition ECM if this flag is provided")

    args = parser.parse_args(argv)
    demo_number = args.demo_number
    reposition_ecm = args.reposition_ecm

    # Configuration dictionary
    exp_initialiser_config_dict = {"parent_frames": ["Cart", "ECM_ref", "ECM_ref"],
                   "child_frames": ["ECM_ref", "PSM1_ref", "PSM2_ref"],
                   "arm_names": ["ECM", "PSM1", "PSM2"],
                   "transform_lookup_wait_time": 1.0,
                   "sleep_time_between_moves": 1.0,
                   "ros_freq": 10.0,
                   "reposition_ecm": reposition_ecm
    }

    ral = crtk.ral('replay_exp')
    # Create ExperimentInitializer object and run the initialization
    rospy.loginfo("INITIALIZING EXPERIMENT BEFORE REPLAYING")

    initializer = ExperimentInitializer(ral,exp_initialiser_config_dict)

    initialized_exp_successfully = initializer.run()

    if initialized_exp_successfully:
        for t in range(2,-1,-1):
            if rospy.is_shutdown():
                rospy.loginfo("Shutting Down")
                break
            rospy.sleep(1)  # Sleep for a short duration to allow the initialization to complete
            rospy.loginfo(f"REPLAYING EXPERIMENT in {t}")



        csv_file = f"/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Object-Transfer/Demo{demo_number}/data.csv"

        replay_exp_config_dict = {"csv_file": csv_file,
                    "arm_names": ["PSM1", "PSM2"],
                    "ros_freq": 30,
                        "arm_objs":initializer.arm_objs,
                        "initial_joint_state_dicrepancy_tolerance": 0.5 # Used to check if the initial measured joint state is too far from the target joint state from the csv file
                        }  # Specify the arm name if needed

        replay_exp = ReplayExperiment(ral, replay_exp_config_dict)
        replay_exp.run()

