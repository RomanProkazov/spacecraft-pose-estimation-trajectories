import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


# Load the images
image_dir = Path('charuco_1280_720_cv42')
img_names = [image for image in image_dir.glob('*.png')]
img_names = sorted(img_names) # , key=lambda x: int(x.stem.split('_')[-1]))

with open('camera_calibration_1280_720.json') as f:
    data = json.load(f)
camera_matrix = np.array(data['camera_matrix'])
dist_coefs = np.array(data['dist'])
rvecs = np.array(data['rvecs'])
tvecs = np.array(data['tvecs'])


_3d_corners = np.float32([[0,0,0], [0,100,0], [100,100,0], [100,0,0],
                           [0,0,-100],[0,100,-100],[100,100,-100],[100,0,-100]])


image_index=1
cube_corners_2d,_ = cv2.projectPoints(_3d_corners,rvecs[image_index],tvecs[image_index],camera_matrix,dist_coefs) 
cube_corners_2d = np.round(cube_corners_2d).astype(np.int64)
print(cube_corners_2d,0) 


img=cv2.imread(img_names[image_index]) 

red=(0,0,255) #red (in BGR)
blue=(255,0,0) #blue (in BGR)
green=(0,255,0) #green (in BGR)
line_width=2

#first draw the base in red
cv2.line(img, tuple(cube_corners_2d[0][0]), tuple(cube_corners_2d[1][0]),red,line_width)
cv2.line(img, tuple(cube_corners_2d[1][0]), tuple(cube_corners_2d[2][0]),red,line_width)
cv2.line(img, tuple(cube_corners_2d[2][0]), tuple(cube_corners_2d[3][0]),red,line_width)
cv2.line(img, tuple(cube_corners_2d[3][0]), tuple(cube_corners_2d[0][0]),red,line_width)

#now draw the pillars
cv2.line(img, tuple(cube_corners_2d[0][0]), tuple(cube_corners_2d[4][0]),blue,line_width)
cv2.line(img, tuple(cube_corners_2d[1][0]), tuple(cube_corners_2d[5][0]),blue,line_width)
cv2.line(img, tuple(cube_corners_2d[2][0]), tuple(cube_corners_2d[6][0]),blue,line_width)
cv2.line(img, tuple(cube_corners_2d[3][0]), tuple(cube_corners_2d[7][0]),blue,line_width)

#finally draw the top
cv2.line(img, tuple(cube_corners_2d[4][0]), tuple(cube_corners_2d[5][0]),green,line_width)
cv2.line(img, tuple(cube_corners_2d[5][0]), tuple(cube_corners_2d[6][0]),green,line_width)
cv2.line(img, tuple(cube_corners_2d[6][0]), tuple(cube_corners_2d[7][0]),green,line_width)
cv2.line(img, tuple(cube_corners_2d[7][0]), tuple(cube_corners_2d[4][0]),green,line_width)
    
plt.imshow(img[...,::-1])
plt.show()