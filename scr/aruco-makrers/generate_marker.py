import cv2
import numpy as np
from cv2 import aruco


aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
marker = aruco.generateImageMarker(aruco_dict, 23, 200)
cv2.imwrite("aruco_marker.png", marker)