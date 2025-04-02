#!/usr/bin/env python


'''
Author: Chetan Reddy Narayanaswamy

This script is used to control the arms of the da vinci using the keyboard input
Left, Right, Up, Down, Page Up, Page Down: Controls Arm 1 in Cartesian Space (X,Y,Z axes)
W,A,S,D,Q,E: Controls Arm 2 in Cartisian Space
1,2,3,4,5,6: Increases the corresponding joint value
Shift + (1,2,3,4,5,6): Decreases the correspoinding value
7 : Increases the Jaw Opening
Shift + 7 : Decreases the Jaw Opening

'''

# Import required libraries
import argparse
import sys
import time
import crtk
import dvrk
import numpy
import PyKDL
from pynput import keyboard  # For keyboard input

# Teleoperation class
class teleop_application:

    # Configuration
    def __init__(self, ral, config_dict):
        
        self.ral = ral
        self.expected_interval = config_dict['expected_interval']
        self.arm1_name = config_dict['arm1_name']
        self.arm2_name = config_dict['arm2_name']
        self.num_arms = config_dict['num_arms']
        self.step_size = config_dict['step_size'] #  Step size for movement (in meters)
        self.joint_step_size = config_dict['joint_step_size'] # Step size for joint movement
        self.control_type = config_dict['control_type']
        self.arm1 = dvrk.psm(ral, self.arm1_name)

        if self.num_arms == 2:
            self.arm2 = dvrk.psm(ral, self.arm2_name)
            
        self.running = False
        self.dx1 = 0  # X-axis movement for arm1
        self.dy1 = 0  # Y-axis movement for arm1
        self.dz1 = 0  # Z-axis movement for arm1
        self.dx2 = 0  # X-axis movement for arm2
        self.dy2 = 0  # Y-axis movement for arm2
        self.dz2 = 0  # Z-axis movement for arm2
        self.joint_deltas = numpy.zeros(6)  # Assuming the robot has 6 joints
        self.jaw_delta = 0  # Jaw movement

        print('Initialising teleop system for {} and {}'.format(self.arm1_name, self.arm2_name))
        print('Using Control Type: {}'.format(self.control_type))

    # Homing the arms
    def home(self):
        self.arm1.check_connections()

        print('Starting enable')
        if not self.arm1.enable(10):
            sys.exit('Failed to enable within 10 seconds')
        print('Starting home')
        if not self.arm1.home(10):
            sys.exit('Failed to home within 10 seconds')

        if self.num_arms == 2: # If there is a second arm activated
            self.arm2.check_connections()
            
            if not self.arm2.enable(10):
                sys.exit('Failed to enable within 10 seconds')

            if not self.arm2.home(10):
                sys.exit('Failed to home within 10 seconds')

        print('Homing complete')

    # Move the arm's tip incrementally in Cartesian coordinates
    def move_cartesian(self):

        current_pose1 = self.arm1.setpoint_cp() # type ='PyKDL.Frame'
        goal1 = PyKDL.Frame()
        goal1.p = current_pose1.p
        goal1.M = current_pose1.M
        goal1.p[0] += self.dx1 * self.step_size
        goal1.p[1] += self.dy1 * self.step_size
        goal1.p[2] += self.dz1 * self.step_size

        if self.control_type == 's':
            self.arm1.servo_cp(goal1)
        elif self.control_type == 'm':
            self.arm1.move_cp(goal1).wait()

        if self.num_arms == 2:
            current_pose2 = self.arm2.setpoint_cp()
            goal2 = PyKDL.Frame()
            goal2.p = current_pose2.p
            goal2.M = current_pose2.M
            goal2.p[0] += self.dx2 * self.step_size
            goal2.p[1] += self.dy2 * self.step_size
            goal2.p[2] += self.dz2 * self.step_size

            if self.control_type == 's':
                self.arm2.servo_cp(goal2)
            elif self.control_type == 'm':
                self.arm2.move_cp(goal2).wait()

    def move_joint(self):
        # Move joints
        current_joint_positions = self.arm1.setpoint_jp()
        current_jaw_position = self.arm1.jaw.setpoint_jp()

        new_joint_positions = current_joint_positions + self.joint_deltas
        new_jaw_position = current_jaw_position + self.jaw_delta

        self.arm1.servo_jp(new_joint_positions)
        self.arm1.jaw.servo_jp(new_jaw_position)


    # Keyboard event handlers
    def on_press(self, key):
        try:
            if key == keyboard.Key.esc:
                self.running = False  # Set running to False to exit the main loop
                return False  # Stop the listener
            # Control PSM1 with arrow keys
            if key == keyboard.Key.up:
                print('Pressed Up: Moving Arm1 Along +y axis')
                self.dy1 = 1
            elif key == keyboard.Key.down:
                print('Pressed Down: Moving Arm1 Along -y axis')
                self.dy1 = -1
            elif key == keyboard.Key.left:
                print('Pressed Left: Moving Arm1 Along -x axis')
                self.dx1 = -1
            elif key == keyboard.Key.right:
                print('Pressed Right: Moving Arm1 Along +x axis')
                self.dx1 = 1
            elif key == keyboard.Key.page_up:
                print('Pressed PageUp: Moving Arm1 Along +z axis')
                self.dz1 = 1
            elif key == keyboard.Key.page_down:
                print('Pressed PageDown: Moving Arm1 Along -z axis')
                self.dz1 = -1


            # Jaw control
            if hasattr(key, 'char'):
                if key.char == '[':
                    print('Pressed [: Opening Jaw of Arm1')
                    self.arm1.jaw.open()#.wait(is_busy = True)
                    # self.arm.insert_jp(0.1).wait()
                elif key.char == ']':
                    print('Pressed ]: Closing Jaw of Arm1')
                    self.arm1.jaw.close()#.wait(is_busy = True)
                    
                # Joint control
                elif key.char == '1':
                    print('Pressed 1: Increasing joint 1')
                    self.joint_deltas[0] = self.joint_step_size  # Adjust the increment value as needed
                elif key.char == '2':
                    print('Pressed 2: Increasing joint 2')
                    self.joint_deltas[1] = self.joint_step_size
                elif key.char == '3':
                    print('Pressed 3: Increasing joint 3')
                    self.joint_deltas[2] = self.joint_step_size
                elif key.char == '4':
                    print('Pressed 4: Increasing joint 4')
                    self.joint_deltas[3] = self.joint_step_size
                elif key.char == '5':
                    print('Pressed 5: Increasing joint 5')
                    self.joint_deltas[4] = self.joint_step_size
                elif key.char == '6':
                    print('Pressed 6: Increasing joint 6')
                    self.joint_deltas[5] = self.joint_step_size
                elif key.char == '7':
                    print('Pressed 7: Increasing joint 7')
                    self.jaw_delta = self.joint_step_size
                elif key.char == '!':
                    print('Pressed Shift+1: Decreasing joint 1')
                    self.joint_deltas[0] = -self.joint_step_size
                elif key.char == '@':
                    print('Pressed Shift+2: Decreasing joint 2')
                    self.joint_deltas[1] = -self.joint_step_size
                elif key.char == '#':
                    print('Pressed Shift+3: Decreasing joint 3')
                    self.joint_deltas[2] = -self.joint_step_size
                elif key.char == '$':
                    print('Pressed Shift+4: ecreasing joint 4')
                    self.joint_deltas[3] = -self.joint_step_size
                elif key.char == '%':
                    print('Pressed Shift+5: Decreasing joint 5')
                    self.joint_deltas[4] = -self.joint_step_size
                elif key.char == '^':
                    print('Pressed Shift+6: Decreasing joint 6')
                    self.joint_deltas[5] = -self.joint_step_size
                elif key.char == '&':
                    print('Pressed Shift+7: Decreasing joint 7')
                    self.jaw_delta = -self.joint_step_size



                if self.num_arms == 2:
                    # Control arm2 with WASD keys
                    if key.char == 'w':
                        print('Pressed W: Moving Arm2 Along +y axis')
                        self.dy2 = 1
                    elif key.char == 's':
                        print('Pressed S: Moving Arm2 Along -y axis')
                        self.dy2 = -1
                    elif key.char == 'a':
                        print('Pressed A: Moving Arm2 Along -x axis')
                        self.dx2 = -1
                    elif key.char == 'd':
                        print('Pressed D: Moving Arm2 Along +x axis')
                        self.dx2 = 1
                    elif key.char == 'q':
                        print('Pressed Q: Moving Arm2 Along +z axis')
                        self.dz2 = 1
                    elif key.char == 'e':
                        print('Pressed E: Moving Arm2 Along -z axis')
                        self.dz2 = -1
                

        except Exception as e:
            print('ERROR WHEN Pressing KEY | Key: {} | Exception: {}'.format(key,e))

    def on_release(self, key):
        try:
            # Control PSM1 with arrow keys
            if key in [keyboard.Key.up, keyboard.Key.down]:
                self.dy1 = 0
            elif key in [keyboard.Key.left, keyboard.Key.right]:
                self.dx1 = 0
            elif key in [keyboard.Key.page_up, keyboard.Key.page_down]:
                self.dz1 = 0

            if hasattr(key, 'char'):
                if key.char in '1234567!@#$%^&': # Joint Control
                    self.joint_deltas.fill(0)
                    self.jaw_delta = 0

            if self.num_arms == 2:
                if hasattr(key, 'char'):

                    # Control arm2 with WASD keys
                    if key.char in ['w', 's']:
                        self.dy2 = 0
                    elif key.char in ['a', 'd']:
                        self.dx2 = 0
                    elif key.char in ['q', 'e']:
                        self.dz2 = 0

        except Exception as e:
            print('ERROR WHEN RELEASING KEY | Key: {} | Exception: {}'.format(key,e))

    # Main method
    def run(self):
        self.home()
        self.running = True
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        while self.running:
            self.move_cartesian()
            self.move_joint()
            time.sleep(self.expected_interval)

if __name__ == '__main__':
    # Extract ROS arguments (e.g. __ns:= for namespace)
    argv = crtk.ral.parse_argv(sys.argv[1:])  # Skip argv[0], script name

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a1', '--arm1', type=str, default='PSM1',
                        choices=['ECM', 'MTML', 'MTMR', 'PSM1', 'PSM2', 'PSM3'],
                        help='arm1 name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace')
    parser.add_argument('-a2', '--arm2', type=str, default='PSM2',
                        choices=['ECM', 'MTML', 'MTMR', 'PSM1', 'PSM2', 'PSM3'],
                        help='arm2 name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace')
    parser.add_argument('-n', '--num_arms', type=float, default=2)
    parser.add_argument('-i', '--interval', type=float, default=0.005,
                        help='expected interval in seconds between messages sent by the device')
    parser.add_argument('-s', '--step_size', type=float, default=0.0001,
                        help='step size for movement in meters')
    parser.add_argument('-j', '--joint_step_size', type=float, default=0.005,
                        help='step size for movement in meters')
    parser.add_argument('-c', '--control_type', type=str, default='s',
                        help='s - servo_cp, m - move_cp, i - interpolate_cp')
    args = parser.parse_args(argv)


    config_dict = {'arm1_name':args.arm1,
                   'arm2_name':args.arm2,
                   'num_arms':args.num_arms,
                   'expected_interval':args.interval,
                   'step_size':args.step_size,
                   'joint_step_size':args.joint_step_size,
                   'control_type':args.control_type}

    ral = crtk.ral('teleop_keyboard')
    application = teleop_application(ral, config_dict)
    ral.spin_and_execute(application.run)