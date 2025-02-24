#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from omni_msgs.msg import OmniButtonEvent

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
        rospy.Subscriber('phantom/button', OmniButtonEvent, self.button_callback)
    
        self.phantom_orientation = None
        self.phantom_position = None
        self.current_jaw_pose = None
        
        self.initalised_teleop = False
        self.scale = 1.0
        self.jaw_step_size = 0.1

    def pose_callback(self, msg):
        self.phantom_orientation = msg.pose.orientation # quaternion
        self.phantom_position = msg.pose.position # position

    def button_callback(self, msg):
        if msg.grey_button == 1:
            self.current_jaw_pose += self.jaw_step_size
            print("Opening Jaw")
        elif msg.white_button == 1:
            self.current_jaw_pose -= self.jaw_step_size
            print("Closing Jaw")
        
        # self.arm.jaw.move_jp(self.current_jaw_pose)



    def initialise_phantom_teleop(self):
        '''
        This function is called only once whenever the teleop starts
        - It saves the initial position (x,y,z) of the arm and phantom
        - It changes the orientation of the arm to match that of the phantom omni
        - We use arm.move_cp instead of servo_cp here
        '''
        if self.phantom_position is None or self.phantom_orientation is None:
            print('No Phantom Omni Pose received yet')
            return
        
        rospy.loginfo('Initialising Phantom Teleop')
        goal = PyKDL.Frame()
        self.initial_arm_pose = self.arm.setpoint_cp()
        self.current_jaw_pose = self.arm.jaw.setpoint_jp()

        self.initial_arm_position = self.initial_arm_pose.p

        self.initial_phantom_position = PyKDL.Vector(self.phantom_position.x, self.phantom_position.y, self.phantom_position.z)

        # Aligning the orientation of the arm with the phantom omni
        goal.p = self.initial_arm_pose.p # Keep the (x,y,z) position the same
        q = [self.phantom_orientation.x, self.phantom_orientation.y, self.phantom_orientation.z, self.phantom_orientation.w]
        goal.M = PyKDL.Rotation.Quaternion(q[0],q[1],q[2],q[3])

        # Move the arm to the goal pose
        print('Using move_cp to align the arm: Takes less than a minute | Please wait')
        self.arm.move_cp(goal).wait(True)

        self.initalised_teleop = True
        print('Initialisation complete')


    def move_cartesian(self):
        if self.phantom_position is None or self.phantom_orientation is None:
            print('No Phantom Omni Pose received yet')
            return

        # Create a PyKDL.Frame object for the goal pose
        goal = PyKDL.Frame()

        # Set the position
        phantom_position_pykdl = PyKDL.Vector(self.phantom_position.x, self.phantom_position.y, self.phantom_position.z)
        phantom_translation = phantom_position_pykdl - self.initial_phantom_position

        goal.p = self.initial_arm_position + self.scale*phantom_translation
        # Convert the quaternion to a PyKDL.Rotation
        q = [self.phantom_orientation.x, self.phantom_orientation.y, self.phantom_orientation.z, self.phantom_orientation.w]
        goal.M = PyKDL.Rotation.Quaternion(q[0],q[1],q[2],q[3])

        # Move the arm to the goal pose
        self.arm.move_cp(goal).wait(True)
        # self.arm.servo_cp(goal)

        #.wait()

    def run(self):
        while not self.initalised_teleop:
            self.initialise_phantom_teleop()
            rospy.sleep(0.1)

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