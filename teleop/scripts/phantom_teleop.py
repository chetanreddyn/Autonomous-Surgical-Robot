#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
import dvrk
import numpy
import crtk
import sys
import argparse

class MimicJoints:
    def __init__(self, ral, arm_name):
        self.arm = dvrk.psm(ral, arm_name)

        # Subscribe to the /joint_states topic
        rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)

    def joint_states_callback(self, msg):
        current_joint_positions = self.arm.setpoint_jp()

        # Access the 4th and 5th joint values (yaw and pitch)
        yaw = msg.position[3]
        pitch = msg.position[4]

        # Set the corresponding joints of the dVRK robot
        
        # current_joint_positions[3] = yaw
        current_joint_positions[4] = pitch
        # current_joint_positions[5] = msg.position[5]

        # Move the dVRK robot joints
        self.arm.move_jp(current_joint_positions)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    # Extract ROS arguments (e.g. __ns:= for namespace)
    argv = crtk.ral.parse_argv(sys.argv[1:])  # Skip argv[0], script name

    # Initialize the MimicJoints class with the name of the dVRK arm
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arm', type=str, default='PSM1',
                        choices=['ECM', 'MTML', 'MTMR', 'PSM1', 'PSM2', 'PSM3'],
                        help='arm name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace')
    args = parser.parse_args(argv)

    ral = crtk.ral('mimic_joints')
    mimic_joints = MimicJoints(ral, args.arm)
    ral.spin_and_execute(mimic_joints.run)