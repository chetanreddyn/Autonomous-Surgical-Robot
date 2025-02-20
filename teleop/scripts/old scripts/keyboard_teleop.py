#!/usr/bin/env python

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
    def __init__(self, ral, arm_name, expected_interval,step_size):
        print('Configuring teleop system for {}'.format(arm_name))
        self.ral = ral
        self.expected_interval = expected_interval
        # self.arm = dvrk.arm(ral=ral, arm_name=arm_name, expected_interval=expected_interval)
        self.arm = dvrk.psm(ral, arm_name)
        self.step_size = step_size  # Step size for movement (in meters)
        self.running = False
        self.dx = 0  # X-axis movement
        self.dy = 0  # Y-axis movement

    # Homing the arm
    def home(self):
        self.arm.check_connections()
        print('Starting enable')
        if not self.arm.enable(10):
            sys.exit('Failed to enable within 10 seconds')
        print('Starting home')
        if not self.arm.home(10):
            sys.exit('Failed to home within 10 seconds')
        print('Homing complete')

    # Move the arm's tip incrementally
    def move_tip(self):
        # Get current cartesian position
        current_pose = self.arm.setpoint_cp()
        goal = PyKDL.Frame()
        goal.p = current_pose.p
        goal.M = current_pose.M

        # Update goal position
        goal.p[0] += self.dx  # X-axis
        goal.p[1] += self.dy  # Y-axis

        # Send new goal to the arm
        self.arm.servo_cp(goal)

    # Keyboard event handlers
    def on_press(self, key):
        try:
            if key == keyboard.Key.up:
                self.dy = self.step_size  # Move up (+Y)
            elif key == keyboard.Key.down:
                self.dy = -self.step_size  # Move down (-Y)
            elif key == keyboard.Key.left:
                self.dx = -self.step_size  # Move left (-X)
            elif key == keyboard.Key.right:
                self.dx = self.step_size  # Move right (+X)

        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if key == keyboard.Key.up or key == keyboard.Key.down:
                self.dy = 0  # Stop Y-axis movement
            elif key == keyboard.Key.left or key == keyboard.Key.right:
                self.dx = 0  # Stop X-axis movement
            elif key == keyboard.KeyCode.from_char('q'):  # Quit on 'q'
                print('Exiting teleoperation.')
                self.running = False
                return False  # Stop the listener
        except AttributeError:
            pass

    # Main teleoperation loop
    def run_teleop(self):
        print('Starting teleoperation. Use arrow keys to move the arm. Press "q" to quit.')
        self.running = True

        # Start keyboard listener
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        # Main control loop
        while self.running:
            self.move_tip()  # Update arm position
            time.sleep(self.expected_interval)  # Sleep to avoid excessive CPU usage

        # Stop the listener when done
        listener.stop()

    # Main method
    def run(self):
        self.home()
        self.run_teleop()

# Main execution
if __name__ == '__main__':
    # Extract ROS arguments (e.g., __ns:= for namespace)
    argv = crtk.ral.parse_argv(sys.argv[1:])  # Skip argv[0], script name

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arm', type=str, required=True,
                        choices=['ECM', 'MTML', 'MTMR', 'PSM1', 'PSM2', 'PSM3'],
                        help='Arm name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace')
    parser.add_argument('-i', '--interval', type=float, default=0.01,
                        help='Expected interval in seconds between messages sent by the device')
    parser.add_argument('-s','--stepsize',type=float,default=0.001,
                        help='Every Time the Keyboard key is pressed, the arm is moved by "stepsize"')
    args = parser.parse_args(argv)

    # Initialize ROS abstraction layer and run application
    ral = crtk.ral('dvrk_teleop')
    application = teleop_application(ral, args.arm, args.interval,args.stepsize)
    ral.spin_and_execute(application.run)