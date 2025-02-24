#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

import dvrk
import numpy
import crtk
import sys
import argparse
import PyKDL


class MimicPose:
    def __init__(self, ral, arm_name):
        self.arm = dvrk.psm(ral, arm_name)

        # Subscribe to the /joint_states topic
        rospy.Subscriber('phantom/pose', PoseStamped, self.pose_callback)
    
        self.phantom_orientation = None
        self.phantom_pose = None

    def pose_callback(self, msg):
        self.phantom_orientation = msg.pose.orientation
        self.phantom_pose = msg.pose.position

    def move_cartesian(self):
        if self.phantom_pose is None or self.phantom_orientation is None:
            return

        # Create a PyKDL.Frame object for the goal pose
        goal = PyKDL.Frame()
        current_pose = self.arm.setpoint_cp()

        # Set the position
        # goal.p[0] = self.phantom_pose.x
        # goal.p[1] = self.phantom_pose.y
        # goal.p[2] = self.phantom_pose.z
        goal.p = current_pose.p
        print("hereeeee")
        # Convert the quaternion to a PyKDL.Rotation
        quaternion = [self.phantom_orientation.x, self.phantom_orientation.y, self.phantom_orientation.z, self.phantom_orientation.w]
        goal.M = PyKDL.Rotation.Quaternion(*quaternion)
        # goal.M = current_pose.M
        # Move the arm to the goal pose
        self.arm.move_cp(goal).wait()

    def run(self):

        while not rospy.is_shutdown():
            self.move_cartesian()
            rospy.sleep(0.1)
        # rospy.spin()

if __name__ == '__main__':
    # Extract ROS arguments (e.g. __ns:= for namespace)
    argv = crtk.ral.parse_argv(sys.argv[1:])  # Skip argv[0], script name

    # Initialize the MimicJoints class with the name of the dVRK arm
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arm', type=str, default='PSM1',
                        choices=['ECM', 'MTML', 'MTMR', 'PSM1', 'PSM2', 'PSM3'],
                        help='arm name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace')
    args = parser.parse_args(argv)

    ral = crtk.ral('mimic_pose')
    mimic_pose = MimicPose(ral, args.arm)
    ral.spin_and_execute(mimic_pose.run)