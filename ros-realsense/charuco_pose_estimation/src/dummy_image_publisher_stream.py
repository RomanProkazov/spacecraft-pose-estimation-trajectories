#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

def create_dummy_image():
    # Create a blank 1280x720 blue image with markers
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:,:,0] = 200  # Blue background
    
    # Add ArUco markers (white squares with black borders)
    marker_size = 100
    positions = [(100,100), (400,100), (700,400)]
        
    return img

def publish_dummy_data():
    rospy.init_node('dummy_publisher')
    bridge = CvBridge()
    
    # Publishers
    image_pub = rospy.Publisher('/camera/color/image_raw', Image, queue_size=1)
    pose_pub = rospy.Publisher('/mockup_in_cam/pose', PoseStamped, queue_size=1)
    
    rate = rospy.Rate(10)  # 10Hz
    
    while not rospy.is_shutdown():
        # Publish image
        dummy_img = create_dummy_image()
        img_msg = bridge.cv2_to_imgmsg(dummy_img, "bgr8")
        img_msg.header.stamp = rospy.Time.now()
        img_msg.header.frame_id = "camera_frame"  # Must match camera frame
        image_pub.publish(img_msg)
        
        # Publish pose (moving in a circle)
        t = rospy.Time.now().to_sec()
        
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "camera_frame"  # Critical fix
        
        # Circular motion (1m radius)
        pose.pose.position.x = 0.5 * np.sin(t)
        pose.pose.position.y = 0.5 * np.cos(t) 
        pose.pose.position.z = 1.5  # 1.5m in front
        
        # Slowly rotating
        angle = t * 0.3  # 0.3 rad/sec
        pose.pose.orientation.w = np.cos(angle/2)
        pose.pose.orientation.x = 0
        pose.pose.orientation.y = 0
        pose.pose.orientation.z = np.sin(angle/2)
        
        pose_pub.publish(pose)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_dummy_data()
    except rospy.ROSInterruptException:
        pass