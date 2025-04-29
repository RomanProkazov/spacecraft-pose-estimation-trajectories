import PySpin
import cv2
import numpy as np
import time
import os
import screeninfo


# Initialize camera
system = PySpin.System.GetInstance()
cam_list = system.GetCameras()

if cam_list.GetSize() == 0:
    cam_list.Clear()
    system.ReleaseInstance()
    exit()
    