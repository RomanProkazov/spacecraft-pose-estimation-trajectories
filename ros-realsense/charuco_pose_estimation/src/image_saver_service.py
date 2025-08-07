#!/usr/bin/env python3
import rospy
import cv2
import os
from datetime import datetime
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from charuco_pose_estimation.srv import ImageSaver, ImageSaverResponse

class ImageSaverService:
    def __init__(self):
        # Initialize node
        rospy.init_node('image_saver_service')
        
        # Setup directories
        self.output_dir = os.path.expanduser("~/saved_images")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        
        # Publishers
        self.annotated_pub = rospy.Publisher('/annotated_image', Image, queue_size=1)
        
        # Service
        self.service = rospy.Service('save_image_service', ImageSaver, self.handle_request)
        
        # Variables
        self.latest_image = None
        self.latest_image_timestamp = None
        
        rospy.loginfo("Image Saver Service Ready")
    
    def image_callback(self, msg):
        """Store the latest image"""
        self.latest_image = msg
        self.latest_image_timestamp = msg.header.stamp
    
    def handle_request(self, req):
        """Service handler"""
        response = ImageSaverResponse()
        
        try:
            if req.save_image.data:  # Changed from process_latest_image to save_image
                if self.latest_image is None:
                    response.success = False
                    response.message = "No image available"
                    response.image_path = ""
                    return response
                
                # Process image
                cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
                
                # Save image
                timestamp = self.latest_image_timestamp.to_sec()
                time_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S_%f")
                image_path = os.path.join(self.output_dir, f"image_{time_str}.jpg")
                cv2.imwrite(image_path, cv_image)
                
                # Publish the same image to annotated topic
                self.annotated_pub.publish(self.latest_image)
                
                response.success = True
                response.message = "Image saved successfully"
                response.image_path = image_path
            else:
                response.success = False
                response.message = "Processing not requested"
                response.image_path = ""
            
            return response
            
        except Exception as e:
            response.success = False
            response.message = str(e)
            response.image_path = ""
            return response

if __name__ == "__main__":
    try:
        service = ImageSaverService()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass    