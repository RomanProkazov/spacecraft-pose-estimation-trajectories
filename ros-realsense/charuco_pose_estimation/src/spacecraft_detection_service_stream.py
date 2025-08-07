#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import json
import os
import threading
from datetime import datetime
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import Bool
from charuco_pose_estimation.srv import SpacecraftDetection, SpacecraftDetectionResponse
import tf.transformations

class SpacecraftVisualizer:
    def __init__(self):
        rospy.init_node('spacecraft_detection_service')
        
        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_image_timestamp = None
        self.latest_pose = None
        
        # Configuration
        self.ARUCO_DICT = cv2.aruco.DICT_6X6_250
        self.OUTPUT_BASE = os.path.expanduser("~/spacecraft_detections")
        self.OUTPUT_IMG_DIR = os.path.join(self.OUTPUT_BASE, "original_images")
        self.OUTPUT_IMGan_DIR = os.path.join(self.OUTPUT_BASE, "annotated_images")
        self.OUTPUT_JSON_DIR = os.path.join(self.OUTPUT_BASE, "json_data")
        
        os.makedirs(self.OUTPUT_IMG_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_IMGan_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_JSON_DIR, exist_ok=True)

        # Load camera parameters
        self.camera_matrix, self.dist_coeffs, self.sat_model = self.load_camera_parameters()
        
        # Publishers
        self.annotated_stream_pub = rospy.Publisher('/annotated_stream', Image, queue_size=1)
        self.snapshot_pub = rospy.Publisher('/spacecraft_detection/annotated_image', Image, queue_size=1)
        
        # Subscribers
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        rospy.Subscriber('/mockup_in_cam/pose', PoseStamped, self.pose_callback)
        
        # Service
        self.service = rospy.Service('detect_spacecraft', SpacecraftDetection, self.handle_request)
        
        # Start continuous processing thread
        self.process_rate = rospy.Rate(10)  # 30 Hz
        self.processing_thread = threading.Thread(target=self.continuous_processing)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        rospy.loginfo("Spacecraft Detection Service Ready with Live Streaming")

    def load_camera_parameters(self):
        CAMERA_SAT_JSON = "/home/roman/catkin_ws/src/charuco_pose_estimation/cam_sat.json"
        with open(CAMERA_SAT_JSON, 'r') as f:
            data = json.load(f)
        return (
            np.array(data['camera_matrix_rs_1280']),
            np.array(data['dist']),
            np.array(data['sat_model_center'])
        )

    def image_callback(self, msg):
        self.latest_image = msg
        self.latest_image_timestamp = msg.header.stamp

    def pose_callback(self, msg):
        self.latest_pose = msg

    def extract_ros_pose(self, pose):
        quaternion = pose.pose.orientation
        quat_xyzw = np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        
        rot_matrix = tf.transformations.quaternion_matrix(quat_xyzw)[:3, :3]
        rvec, _ = cv2.Rodrigues(rot_matrix)
        tvec = np.array([pose.pose.position.x, 
                         pose.pose.position.y,
                         pose.pose.position.z])
        
        return quat_xyzw, rvec, tvec

    def project_keypoints(self, pose):
        quat_xyzw, rvec, tvec = self.extract_ros_pose(pose)
        image_points, _ = cv2.projectPoints(self.sat_model, rvec, tvec, 
                                          self.camera_matrix, self.dist_coeffs)
        return image_points.reshape(-1, 2)

    def process_frame(self, save_output=False):
        if self.latest_image is None:
            rospy.logwarn("No image available")

            return None
        if self.latest_pose is None:
            rospy.logwarn("No pose available")

            return None

        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
            annotated_image = cv_image.copy()
            
            # Project keypoints
            kpts = self.project_keypoints(self.latest_pose)
            for point in kpts:
                x, y = round(point[0]), round(point[1])
                cv2.circle(annotated_image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

            # Detect ArUco markers
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            aruco_dict = cv2.aruco.Dictionary_get(self.ARUCO_DICT)
            parameters = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            
            if ids is not None:
                annotated_image = cv2.aruco.drawDetectedMarkers(annotated_image, corners, ids)
                for i, marker_id in enumerate(ids):
                    center = np.mean(corners[i][0], axis=0).astype(int)
                    cv2.putText(annotated_image, f"ID:{marker_id[0]}", tuple(center),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            if save_output:
                timestamp = self.latest_image_timestamp.to_sec()

                # Save images
                original_filename = f"{self.OUTPUT_IMG_DIR}/image_{timestamp}.jpg"
                annotated_filename = f"{self.OUTPUT_IMGan_DIR}/imagean_{timestamp}.jpg"
                cv2.imwrite(original_filename, cv_image)
                cv2.imwrite(annotated_filename, annotated_image)

                # print(corners)
                
                # Save JSON data
                json_filename = f"{self.OUTPUT_JSON_DIR}/image_{timestamp}.json"
                data = {
                    "timestamp": timestamp,
                    "original_image_path": f"image_{timestamp}.jpg",
                    "translation": [
                        self.latest_pose.pose.position.x,
                        self.latest_pose.pose.position.y,
                        self.latest_pose.pose.position.z
                    ],
                    "pose": [
                        self.latest_pose.pose.orientation.w,
                        self.latest_pose.pose.orientation.x,
                        self.latest_pose.pose.orientation.y,
                        self.latest_pose.pose.orientation.z,
                    ],
                    "projected_keypoints": kpts.tolist(),
                    "detected_markers": [] if ids is None else [int(id[0]) for id in ids]
                }
                
                with open(json_filename, 'w') as f:
                    json.dump(data, f, indent=4)

            return annotated_image

        except Exception as e:
            rospy.logerr(f"Processing error: {str(e)}")
            return None

    def continuous_processing(self):
        while not rospy.is_shutdown():
            annotated_img = self.process_frame(save_output=False)
            if annotated_img is not None:
                try:
                    msg = self.bridge.cv2_to_imgmsg(annotated_img, "bgr8")
                    msg.header.stamp = rospy.Time.now()
                    self.annotated_stream_pub.publish(msg)
                except Exception as e:
                    rospy.logerr(f"Stream publishing error: {str(e)}")
            self.process_rate.sleep()

    def handle_request(self, req):
        response = SpacecraftDetectionResponse()
        
        if req.process_image.data:
            annotated_img = self.process_frame(save_output=True)
            if annotated_img is not None:
                try:
                    # Publish snapshot
                    msg = self.bridge.cv2_to_imgmsg(annotated_img, "bgr8")
                    msg.header.stamp = rospy.Time.now()
                    self.snapshot_pub.publish(msg)
                    
                    response.success = True
                    response.message = "Processed successfully"
                except Exception as e:
                    response.success = False
                    response.message = str(e)
            else:
                response.success = False
                response.message = "Failed to process image"
        else:
            response.success = False
            response.message = "Processing not requested"
        
        response.pose_stamped = self.latest_pose if self.latest_pose else PoseStamped()
        return response

if __name__ == "__main__":
    visualizer = SpacecraftVisualizer()
    rospy.spin()