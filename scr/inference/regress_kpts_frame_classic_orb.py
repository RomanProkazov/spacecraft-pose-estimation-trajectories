import cv2

img_real = "/home/roman/spacecraft-pose-estimation-trajectories/data_real/lux_sat_data_real_v1_nobck/frame_0737.jpg"
img_fake = "/home/roman/spacecraft-pose-estimation-trajectories/data/images_leo_v1/image_10000.jpg"
# Load an image
img = cv2.imread(img_real)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize the ORB detector
orb = cv2.ORB_create()

# Detect keypoints
keypoints = orb.detect(gray, None)
print(len(keypoints))

# Draw the first 50 keypoints on the image
img_with_keypoints = cv2.drawKeypoints(
    img, keypoints[:500], None, color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imshow('Keypoints', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()