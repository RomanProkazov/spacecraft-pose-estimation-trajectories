#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import time

class RealSenseImageViewer:
    def __init__(self):
        self.bridge = CvBridge()
        self.last_saved_time = 0
        self.save_interval = 1.0 
        self.image_counter = 0
        rospy.init_node('realsense_image_viewer', anonymous=True)
        rospy.Subscriber('/camera/color/image_raw', Image, self.callback)

        # Output folder to save images (optional)
        self.save_images = rospy.get_param('~save_images', False)
        self.output_dir = rospy.get_param('~output_dir', '/home/roman/spacecraft-pose-estimation-trajectories/data_real/images')
        os.makedirs(self.output_dir, exist_ok=True)

        self.image_count = 0
        rospy.loginfo("RealSense Image Viewer node started.")
        rospy.spin()

    def callback(self, msg):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CVBridge error: {e}")
            return

        cv2.imshow("RealSense Camera", cv_image)
        key = cv2.waitKey(1)

        if self.save_images:
            filename = os.path.join(self.output_dir, f"frame_{self.image_count:04d}.jpg")
            cv2.imwrite(filename, cv_image)
            self.image_count += 1

        if key == ord('q'):
            rospy.signal_shutdown("Quit by user.")
            cv2.destroyAllWindows()

if __name__ == '__main__':
    RealSenseImageViewer()
