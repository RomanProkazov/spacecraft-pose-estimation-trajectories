#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImageViewer:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_counter = 0
        self.frame_counter = 0
        self.save_every_n_frames = 10  # Save every 10 frames

        self.output_dir = "spacecraft-pose-estimation-trajectories/data_real/images"
        os.makedirs(self.output_dir, exist_ok=True)

        rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        rospy.loginfo("ImageViewer initialized. Showing and saving every %d frames.", self.save_every_n_frames)

    def callback(self, msg):
        self.frame_counter += 1

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow("Camera View", cv_image)
            cv2.waitKey(1)

            # Save every Nth frame
            if self.frame_counter % self.save_every_n_frames == 0:
                filename = os.path.join(self.output_dir, f"frame_{self.image_counter:04d}.jpg")
                cv2.imwrite(filename, cv_image)
                rospy.loginfo("Saved image: %s", filename)
                self.image_counter += 1

        except Exception as e:
            rospy.logerr("Error processing image: %s", e)

if __name__ == '__main__':
    rospy.init_node('image_viewer', anonymous=True)
    viewer = ImageViewer()
    rospy.spin()
    cv2.destroyAllWindows()
