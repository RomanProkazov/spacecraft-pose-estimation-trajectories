#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo

def camera_info_callback(msg):
    rospy.loginfo(f"Image width: {msg.width}, height: {msg.height}")
    rospy.loginfo(f"Camera matrix K: {msg.K}")
    rospy.signal_shutdown("Got one message.")

def main():
    rospy.init_node('camera_info_listener', anonymous=True)
    rospy.Subscriber('/camera/color/image_raw/info', CameraInfo, camera_info_callback)
    rospy.spin()

if __name__ == '__main__':
    main()