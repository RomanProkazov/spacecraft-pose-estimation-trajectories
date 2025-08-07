#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from cv_bridge import CvBridge

def create_dummy_image():
    # Create a blank 1280x720 blue image (OpenCV BGR format)
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:,:,0] = 255  # Set blue channel to max (OpenCV uses BGR)
    
    # # Add some dummy ArUco markers (white squares)
    # img[100:200, 100:200] = 255  # Marker 1
    # img[100:200, 300:400] = 255  # Marker 2
    # img[300:400, 500:600] = 255  # Marker 3
    
    return img

def publish_dummy_data():
    rospy.init_node('dummy_publisher')
    bridge = CvBridge()
    
    # Create publishers
    image_pub = rospy.Publisher('/camera/color/image_raw', Image, queue_size=1)
    pose_pub = rospy.Publisher('/mockup_in_cam/pose', PoseStamped, queue_size=1)
    
    rate = rospy.Rate(10)  # 10 Hz
    
    while not rospy.is_shutdown():
        # Create and publish dummy image
        dummy_img = create_dummy_image()
        img_msg = bridge.cv2_to_imgmsg(dummy_img, encoding="bgr8")
        img_msg.header.stamp = rospy.Time.now()
        img_msg.header.frame_id = "camera_color_optical_frame"
        image_pub.publish(img_msg)
        
        # Create and publish dummy pose (changing over time)
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "camera_frame"
        
        # Set some dummy position (1 meter in front of camera)
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 1.0
        
        # Set some dummy orientation (no rotation)
        pose.pose.orientation.w = 0.71
        pose.pose.orientation.x = 0.71
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        
        pose_pub.publish(pose)
        
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_dummy_data()
    except rospy.ROSInterruptException:
        pass