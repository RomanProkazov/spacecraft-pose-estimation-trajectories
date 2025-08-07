#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import json
import os
from datetime import datetime
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import Bool
from charuco_pose_estimation.srv import SpacecraftDetection, SpacecraftDetectionResponse
import tf.transformations

# Global variables
bridge = CvBridge()
latest_image = None
latest_image_timestamp = None
latest_pose = None

# Configuration
ARUCO_DICT = cv2.aruco.DICT_6X6_250
OUTPUT_BASE = os.path.expanduser("~/spacecraft_detections")
OUTPUT_IMG_DIR = os.path.join(OUTPUT_BASE, "original_images")
OUTPUT_IMGan_DIR = os.path.join(OUTPUT_BASE, "annotated_images")

OUTPUT_JSON_DIR = os.path.join(OUTPUT_BASE, "json_data")
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMGan_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

CAMERA_SAT_JSON = "/home/roman/catkin_ws/src/charuco_pose_estimation/cam_sat.json"

def load_camera_parameters():
    with open(CAMERA_SAT_JSON, 'r') as f:
        data = json.load(f)
    return (
        np.array(data['camera_matrix_rs_1280']),
        np.array(data['dist']),
        np.array(data['sat_model_center'])
    )

camera_matrix, dist_coeffs, sat_model = load_camera_parameters()

def image_callback(msg):
    global latest_image, latest_image_timestamp
    latest_image = msg
    latest_image_timestamp = msg.header.stamp

def pose_callback(msg):
    global latest_pose
    latest_pose = msg


def extract_ros_pose(pose):
    quaternion = pose.pose.orientation
    quat_xyzw = np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
    rot_matrix = tf.transformations.quaternion_matrix(quat_xyzw)
    print(rot_matrix)
    rvec, _ = cv2.Rodrigues(rot_matrix[:3, :3])
    tvec = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])  
    return quat_xyzw, rvec, tvec


def project_keypoints(pose):
    quat_xyzw, rvec, tvec = extract_ros_pose(pose)
    image_points, _ = cv2.projectPoints(sat_model, rvec, tvec, camera_matrix, dist_coeffs)
    return image_points.reshape(-1, 2)


def process_image():
    global latest_image, latest_pose
    
    if latest_image is None:
        return False, "No image available", None, None
    if latest_pose is None:
        return False, "No pose available", None, None

    try:
        # Convert ROS image to OpenCV
        cv_image = bridge.imgmsg_to_cv2(latest_image, "bgr8")
        timestamp = latest_image_timestamp.to_sec()
        time_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S_%f")
        
        # Save original image (without detections)
        original_filename = f"{OUTPUT_IMG_DIR}/image_{timestamp}.jpg"
        cv2.imwrite(original_filename, cv_image)
        
        # Create annotated image (with detections)
        annotated_image = cv_image.copy()
        
        # Project keypoints
        kpts = project_keypoints(latest_pose)
        for point in kpts:
            x, y = round(point[0]), round(point[1])
            cv2.circle(annotated_image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

        # Detect ArUco markers
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT)
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        if ids is not None:
            annotated_image = cv2.aruco.drawDetectedMarkers(annotated_image, corners, ids)
            for i, marker_id in enumerate(ids):
                center = np.mean(corners[i][0], axis=0).astype(int)
                cv2.putText(annotated_image, f"ID:{marker_id[0]}", tuple(center),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Save annotated image
        annotated_filename = f"{OUTPUT_IMGan_DIR}/imagean_{timestamp}.jpg"
        cv2.imwrite(annotated_filename, annotated_image)

            # After creating annotated_image:
        if annotated_image is not None:
            try:
                print("Debug - Annotated image shape:", annotated_image.shape, "type:", annotated_image.dtype)
                
                # Ensure image is proper type
                if annotated_image.dtype != np.uint8:
                    annotated_image = annotated_image.astype(np.uint8)
                    
                # Convert and publish
                annotated_msg = bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
                annotated_msg.header.stamp = rospy.Time.now()
                annotated_pub.publish(annotated_msg)
                
            except Exception as e:
                rospy.logerr("Image publishing failed: " + str(e))
            
        # Save JSON data
        json_filename = f"{OUTPUT_JSON_DIR}/image_{timestamp}.json"
        # corners = corners[0]
        # print(corners)
        data = {
            "timestamp": timestamp,
            "original_image_path": f"image_{timestamp}.jpg",
            
            "translation": [
                latest_pose.pose.position.x,
                latest_pose.pose.position.y,
                latest_pose.pose.position.z
                 ]
            ,
            "pose": [
                latest_pose.pose.orientation.w,
                latest_pose.pose.orientation.x,
                latest_pose.pose.orientation.y,
                latest_pose.pose.orientation.z,
                 ]
            ,
            "projected_keypoints": kpts.tolist(),
            # "detected_marker_corners": corners.tolist()
            
    
        }
        
        with open(json_filename, 'w') as f:
            json.dump(data, f, indent=4)
        
        return True, "Processed successfully", original_filename, json_filename

    except Exception as e:
        return False, str(e), None, None

def handle_request(req):
    try:
        response = SpacecraftDetectionResponse()
        
        if req.process_image.data:
            success, message, img_path, json_path = process_image()
            
            response.success = success
            response.message = message
            response.pose_stamped = latest_pose if latest_pose else PoseStamped()
            response.image_path = img_path if img_path else ""
            response.json_path = json_path if json_path else ""
        else:
            response.success = False
            response.message = "Processing not requested"
            response.pose_stamped = PoseStamped()
            response.image_path = ""
            response.json_path = ""
        
        return response
        
    except Exception as e:
        rospy.logerr(f"Service error: {str(e)}")
        response = SpacecraftDetectionResponse()
        response.success = False
        response.message = str(e)
        response.pose_stamped = PoseStamped()
        response.image_path = ""
        response.json_path = ""
        return response

if __name__ == "__main__":
    rospy.init_node('spacecraft_detection_service')
    
    rospy.Subscriber('/camera/color/image_raw', Image, image_callback)
    rospy.Subscriber('/mockup_in_cam/pose', PoseStamped, pose_callback)
    
    # Publisher for annotated images only
    annotated_pub = rospy.Publisher('/spacecraft_detection/annotated_image', Image, queue_size=1)
    service = rospy.Service('detect_spacecraft', SpacecraftDetection, handle_request)
    
    rospy.loginfo("Spacecraft Detection Service Ready")
    rospy.spin()