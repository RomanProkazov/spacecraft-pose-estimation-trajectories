import cv2

file_path = "/home/roman/spacecraft-pose-estimation-trajectories/data_real/all_images/frame_0000.jpg"

image = cv2.imread(file_path)

# Display the image
cv2.imshow("Ground Truth Visualization", image)
cv2.waitKey(0)
cv2.destroyAllWindows()