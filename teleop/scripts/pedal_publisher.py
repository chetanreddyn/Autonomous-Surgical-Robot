#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Joy
from pynput import keyboard
import threading

class KeyboardPublisher:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('pedal_publisher', anonymous=True)
        
        # Create a publisher for the Joy message
        self.pub = rospy.Publisher('footpedals/coag', Joy, queue_size=10)
        
        # Initialize the Joy message
        self.joy_msg = Joy()
        self.joy_msg.header.stamp = rospy.Time.now()
        self.joy_msg.buttons = [1]  # Initialize with button not pressed

        # Start the keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        # Start the publishing thread
        self.publish_thread = threading.Thread(target=self.publish_loop)

    def on_press(self, key):
        try:
            if key.char == 'i':
                # Set the button state to 1 when "i" is pressed
                self.joy_msg.buttons[0] = 1
                rospy.loginfo("'i' key pressed")
            elif key.char == 'o':
                # Reset the button state to 0 when "o" is pressed
                self.joy_msg.buttons[0] = 0
                rospy.loginfo("'o' key pressed")
        except AttributeError:
            pass
    
    def on_release(self, key):
        try:
            if key.char in 'oi':
                # Reset the button state to 0 when "i" is released
                self.joy_msg.buttons[0] = 1
                rospy.loginfo("'i' key released")
        except AttributeError:
            pass

    def publish_loop(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            self.joy_msg.header.stamp = rospy.Time.now()
            self.pub.publish(self.joy_msg)
            # print(self.joy_msg)
            rate.sleep()

    def run(self):
        self.publish_thread.start()
        
if __name__ == '__main__':
    try:
        keyboard_publisher = KeyboardPublisher()
        keyboard_publisher.run()
    except rospy.ROSInterruptException:
        pass