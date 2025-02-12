#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState

def callback(data):

    names = data.name
    for i,name in enumerate(names):
        print("{}:{:.3f} |".format(name, data.position[i]*180/3.141),end='')
    print()
    print()
def listener():
    rospy.init_node('phantom_test_listener', anonymous=True)
    rospy.Subscriber("/joint_states", JointState, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()