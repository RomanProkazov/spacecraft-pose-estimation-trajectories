#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import json
from pathlib import Path
import sys
from cv_bridge import CvBridge, CvBridgeError
from charuco_pose_estimation.srv import CharUcoPose, CharUcoPoseResponse
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3, PoseStamped, Point, Quaternion
from std_msgs.msg import Float64, Bool
import tf.transformations  # For quaternion conversion
import threading  # For Lock

# Add project root to path (adjust if needed)
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# ------------------------------
# CONFIGURATION (MODIFY THESE)
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7  # Number of squares vertically
SQUARES_HORIZONTALLY = 5  # Number of squares horizontally
SQUARE_LENGTH = 0.037  # 37mm in meters
MARKER_LENGTH = 0.018  # 18mm in meters
CALIBRATION_JSON = "/home/roman/spacecraft-pose-estimation-trajectories/scr/realsense/camera_calibration_1280_720.json"
# ------------------------------

# Globals
bridge = CvBridge()
image_pub = None  # Will be initialized in main
camera_matrix = None
dist_coeffs = None
dictionary = None
board = None
parameters = None
latest_image = None  # Buffer for the latest image
image_lock = threading.Lock()  # Lock to handle image updates safely
processing = False  # Control flag for continuous processing

def load_camera_parameters(calibration_path):
    """Load camera calibration data from JSON file"""
    with open(calibration_path) as f:
        data = json.load(f)
    camera_matrix = np.array(data['camera_matrix'], dtype=np.float32)
    dist_coeffs = np.array(data['dist'], dtype=np.float32)
    return camera_matrix, dist_coeffs

def rvec_to_quaternion(rvec):
    """Convert rotation vector to quaternion."""
    rmat, _ = cv2.Rodrigues(rvec)
    # Pad rotation matrix to 4x4 for tf.transformations
    rmat_4x4 = np.eye(4)
    rmat_4x4[:3, :3] = rmat
    quaternion = tf.transformations.quaternion_from_matrix(rmat_4x4)
    return Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3])

def image_callback(msg):
    """Callback to update the latest image buffer."""
    global latest_image, image_lock
    with image_lock:  # Use a proper lock for thread safety
        latest_image = msg

def detect_pose(image):
    """Detect Charuco board and estimate pose with enhanced visualization (OpenCV 4.2 compatible)"""
    # Detect markers (OpenCV 4.2 style, no ArucoDetector class)
    corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)
    
    if ids is None:
        rospy.loginfo("No markers detected!")
        output_image = image.copy()
        cv2.putText(output_image, "No markers detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return False, output_image, None, None

    # Interpolate Charuco corners
    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, image, board
    )

    if not retval or charuco_ids is None:
        rospy.loginfo("Charuco corners not found!")
        output_image = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)
        for i in range(len(corners)):
            if ids is not None and i < len(ids):
                center = np.mean(corners[i][0], axis=0).astype(int)
                cv2.putText(output_image, f"ID: {ids[i][0]}", tuple(center),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(output_image, "Charuco corners not found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return False, output_image, None, None

    # Estimate pose
    success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board,
        camera_matrix, dist_coeffs, None, None
    )
    
    # Prepare output image
    output_image = image.copy()
    output_image = cv2.aruco.drawDetectedMarkers(output_image, corners, ids)
    
    if not success:
        rospy.loginfo("Pose estimation failed!")
        cv2.putText(output_image, "Pose estimation failed", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return False, output_image, None, None

    # Draw coordinate axes (OpenCV 4.2 uses drawAxis)
    cv2.aruco.drawAxis(output_image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

    # Draw marker IDs
    for i in range(len(corners)):
        if ids is not None and i < len(ids):
            center = np.mean(corners[i][0], axis=0).astype(int)
            cv2.putText(output_image, f"ID: {ids[i][0]}", tuple(center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return True, output_image, rvec, tvec

def estimate_pose_callback(req):
    global bridge, image_pub, latest_image, image_lock, processing

    if not req.process_latest_image.data:
        return CharUcoPoseResponse(success=False, message="Processing not triggered")

    # Wait for a new image if none is available
    while latest_image is None and not rospy.is_shutdown():
        rospy.sleep(0.1)
    
    if latest_image is None:
        return CharUcoPoseResponse(success=False, message="No image available")

    # Process the latest image
    with image_lock:
        cv_image = bridge.imgmsg_to_cv2(latest_image, desired_encoding="bgr8")
        success, output_image, rvec, tvec = detect_pose(cv_image)

    # Publish annotated image (even on failure)
    try:
        image_msg = bridge.cv2_to_imgmsg(output_image, encoding="bgr8")
        image_pub.publish(image_msg)
    except CvBridgeError as e:
        rospy.logerr("Failed to publish annotated image: %s" % e)

    # Prepare response
    resp = CharUcoPoseResponse()
    if success:
        rvec_flat = rvec.flatten()
        tvec_flat = tvec.flatten()
        
        # Populate PoseStamped with capture timestamp from the image
        pose_stamped = PoseStamped()
        pose_stamped.header = latest_image.header  # Use the timestamp and frame_id from the captured image
        pose_stamped.pose.position = Point(x=float(tvec[0][0]), y=float(tvec[1][0]), z=float(tvec[2][0]))
        pose_stamped.pose.orientation = rvec_to_quaternion(rvec)  # Convert rvec to quaternion
        pose_pub.publish(pose_stamped)
        
        resp.pose_stamped = pose_stamped
        resp.success = True
        resp.message = 'Success'
    else:
        resp.success = False
        resp.message = 'Pose estimation failed'

    return resp

if __name__ == "__main__":
    rospy.init_node('charuco_pose_server')
    
    camera_matrix, dist_coeffs = load_camera_parameters(CALIBRATION_JSON)
    dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT)  # OpenCV 4.2 style
    board = cv2.aruco.CharucoBoard_create(
        SQUARES_HORIZONTALLY, SQUARES_VERTICALLY, SQUARE_LENGTH, MARKER_LENGTH, dictionary
    )
    parameters = cv2.aruco.DetectorParameters_create()
    
    # Subscriber for RealSense stream
    rospy.Subscriber('/camera/color/image_raw', Image, image_callback)
    
    # Publisher for annotated image
    image_pub = rospy.Publisher('/annotated_image', Image, queue_size=1)
    pose_pub = rospy.Publisher('/charuco_pose_stream', PoseStamped, queue_size=1, latch=True)
    
    # Service
    s = rospy.Service('estimate_charuco_pose', CharUcoPose, estimate_pose_callback)
    rospy.loginfo('CharUco pose estimation service ready.')
    
    rospy.spin()