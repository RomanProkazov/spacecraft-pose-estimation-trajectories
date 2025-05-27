#!/usr/bin/env python3
import rospy
from std_msgs.msg import Int16

def callback(data):
    num_msg = data.data
    num_msg = num_msg + 1
    print(num_msg)
    
def number_subscriber():

    rospy.init_node('int_subscriber', anonymous=True)

    rospy.Subscriber("number_out", Int16, callback)

    # spin() keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    number_subscriber()
