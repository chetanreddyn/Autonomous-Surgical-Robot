#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import tf.transformations


def callback(data):

    names = data.name
    for i,name in enumerate(names):
        print("{}:{:.3f} |".format(name, data.position[i]*180/3.141),end='')
    print()
    print()

def callback_pose(data):
    # Access the position
    x = data.pose.position.x
    y = data.pose.position.y
    z = data.pose.position.z

    # Access the orientation (quaternion)
    qx = data.pose.orientation.x
    qy = data.pose.orientation.y
    qz = data.pose.orientation.z
    qw = data.pose.orientation.w

    # Convert quaternion to Euler angles
    euler = tf.transformations.euler_from_quaternion([qx, qy, qz, qw])
    alpha = euler[0]*180/3.141
    beta = euler[1]*180/3.141
    gamma = euler[2]*180/3.141

    # Print the position and orientation
    print("p: x={:.3f}, y={:.3f}, z={:.3f} | M: alpha={:.3f}, beta={:.3f}, gamma={:.3f}".format(x, y, z, alpha, beta, gamma))



def listener():
    rospy.init_node('phantom_test_listener', anonymous=True)
    # rospy.Subscriber("/phantom/joint_states", JointState, callback)
    rospy.Subscriber("/phantom/pose", PoseStamped, callback_pose)
    rospy.spin()

if __name__ == '__main__':
    listener()