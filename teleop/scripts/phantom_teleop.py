#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from omni_msgs.msg import OmniButtonEvent
import numpy as np

import dvrk
import numpy
import crtk
import sys
import argparse
import PyKDL

class MimicPose:
    def __init__(self, ral, arm_name,config_dict):
        self.arm = dvrk.psm(ral, arm_name)

        # Subscribe to the /joint_states topic
        self.pose_topic = config_dict['pose_topic']

        rospy.Subscriber(self.pose_topic , PoseStamped, self.pose_callback)
        rospy.Subscriber('phantom/button', OmniButtonEvent, self.button_callback)
    
        self.ros_frequency = config_dict["ros_frequency"]
        self.phantom_orientation = None
        self.phantom_position = None
        self.current_jaw_pose = None

        # The grey button must be pressed once to start
        self.was_in_mono = False
        self.mono = False

        self.enabled = False
        self.open_jaw = False
        self.was_in_open_jaw = False
        self.stylus_pos_received = False
        
        self.scale = config_dict["scale"]
        self.jaw_step_size = config_dict["jaw_step_size"]
        self.jaw_open_angle = config_dict["jaw_open_angle"]
        self.jaw_close_angle = config_dict["jaw_close_angle"]

    def pose_callback(self, msg):
        self.stylus_pos_received = True
        self.phantom_orientation = msg.pose.orientation # quaternion
        self.phantom_position = msg.pose.position # position

    def button_callback(self, msg):

        if msg.grey_button == 1:
            # self.clutch = False
            self.mono = not self.mono # Toggle the mono flag every time the grey button is pressed and RELEASED
            if self.mono:
                rospy.loginfo("Phantom Teleop Switched On")
            else:
                rospy.loginfo("Phantom Teleop Switched Off | Press the Grey Button to Switch on/off")

        # When the grey button is not pressed i.e grey_button=0, The functionality is focused on the white button/jaw
        else:
            if msg.white_button == 1:
                self.open_jaw = True
            
            elif msg.white_button == 0:
                self.open_jaw = False


    def transition_to_enabled(self):
        '''
        This function is called everytime the clutch is pressed
        - It saves the initial position (x,y,z) of the arm and phantom
        - It changes the orientation of the arm to match that of the phantom omni
        - We use arm.move_cp instead of servo_cp here
        '''
        if self.phantom_position is None or self.phantom_orientation is None:
            print('No Phantom Omni Pose received yet, subscribed to: ',self.pose_topic )
            return
        
        rospy.loginfo('Transitioning to ENABLED: Matching Phantom Omni Orientation')
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
        # print('Using move_cp to align the arm: Takes less than a minute | Please wait')
        self.arm.move_cp(goal).wait(True)

        self.enabled = True
        # print('Initialisation complete')


    def move_cartesian(self):
        if self.phantom_position is None or self.phantom_orientation is None and self.enabled:
            print('No Phantom Omni Pose received yet, subscribed to: ',self.pose_topic )
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

        # Reinitialise the phantom position and orientation
        # self.initial_arm_pose = self.arm.setpoint_cp()
        # self.current_jaw_pose = self.arm.jaw.setpoint_jp()
        # self.initial_arm_position = self.initial_arm_pose.p
        # self.initial_phantom_position = PyKDL.Vector(self.phantom_position.x, self.phantom_position.y, self.phantom_position.z)

        # self.arm.servo_cp(goal)

        #.wait()

    def move_jaw(self):
        if self.open_jaw:
            self.arm.jaw.move_jp(np.array([self.jaw_open_angle]))
            self.was_in_open_jaw = True

        elif not self.open_jaw:
            if self.was_in_open_jaw:
                self.arm.jaw.move_jp(np.array([self.jaw_close_angle])) # Closes the Jaw
                self.was_in_open_jaw = False  

    def run(self):
        # while not self.initalised_teleop:
        #     self.initialise_phantom_teleop()
        #     rospy.sleep(0.1)
        rate = rospy.Rate(self.ros_frequency)
        self.transition_to_enabled()
        while not self.stylus_pos_received:
            rospy.loginfo("Waiting for Phantom Omni Pose")
            rospy.sleep(0.1)

        rospy.loginfo("Received! Hold the Pen in Position, Hold the Grey Button Once to Start")
        while not rospy.is_shutdown():
            
            # When mono is pressed
            if self.mono:

                if not self.was_in_mono:
                    self.transition_to_enabled()
                    self.was_in_mono = True 

                self.move_cartesian()
                self.move_jaw()

            # When Mono is not pressed
            elif not self.mono:
                self.was_in_mono = False

  
            rate.sleep()
        # rospy.spin()

if __name__ == '__main__':
    # Extract ROS arguments (e.g. __ns:= for namespace)
    argv = crtk.ral.parse_argv(sys.argv[1:])  # Skip argv[0], script name

    # Initialize the MimicJoints class with the name of the dVRK arm
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arm', type=str, default='PSM1',
                        choices=['ECM', 'MTML', 'MTMR', 'PSM1', 'PSM2', 'PSM3'],
                        help='arm name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace')
    parser.add_argument('-s', '--scale', type=float, default=0.5,
                        help='Scale for Translation')
    parser.add_argument('-j', '--jaw_step_size', type=float, default=0.1,
                    help='Jaw Step Size')
    parser.add_argument('-pt','--pose_topic',type=str,default="/phantom/pose_assistant_perspective",help="The pose to mimic")
    parser.add_argument('-p','--ros_period',type=float,default=0.005,help="Indicates the time period (must match the dvrk_console_json -p flag")
    args = parser.parse_args(argv)

    config_dict = {"scale":args.scale,
                   "jaw_step_size":args.jaw_step_size,
                   "pose_topic":args.pose_topic,
                   "jaw_open_angle":np.pi/3,
                   "jaw_close_angle":0,
                   "ros_frequency":1/args.ros_period}
    
    ral = crtk.ral('mimic_pose')
    mimic_pose = MimicPose(ral, args.arm,config_dict)
    ral.spin_and_execute(mimic_pose.run)