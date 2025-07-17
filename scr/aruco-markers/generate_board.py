import cv2
from cv2 import aruco

# Parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
board = aruco.GridBoard(
        size=(5, 7),  # markersX, markersY
        markerLength=40,
        markerSeparation=10,
        dictionary=aruco_dict
)
img_size = (
    5 * 40 + 4 * 10,  # width = markersX*markerLength + (markersX-1)*markerSeparation
    7 * 40 + 6 * 10   # height = markersY*markerLength + (markersY-1)*markerSeparation
)
board_img = board.generateImage(img_size, marginSize=0)
cv2.imwrite("aruco_board_margin0.png", board_img)