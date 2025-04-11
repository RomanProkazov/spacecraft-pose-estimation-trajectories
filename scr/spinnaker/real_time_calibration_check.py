import PySpin
import cv2
import numpy as np
import json
import screeninfo
import time


def load_calibration_data(calibration_file):
    with open(calibration_file, "r") as f:
        data = json.load(f)
    return np.array(data["camera_matrix"]), np.array(data["distortion_coefficients"])


def draw_cube(img, cube_corners_2d):
    red = (0, 0, 255)  # Red (in BGR)
    blue = (255, 0, 0)  # Blue (in BGR)
    green = (0, 255, 0)  # Green (in BGR)
    line_width = 2

    # Draw the base in red
    cv2.line(img, tuple(cube_corners_2d[0][0]), tuple(cube_corners_2d[1][0]), red, line_width)
    cv2.line(img, tuple(cube_corners_2d[1][0]), tuple(cube_corners_2d[2][0]), red, line_width)
    cv2.line(img, tuple(cube_corners_2d[2][0]), tuple(cube_corners_2d[3][0]), red, line_width)
    cv2.line(img, tuple(cube_corners_2d[3][0]), tuple(cube_corners_2d[0][0]), red, line_width)

    # Draw the pillars in blue
    cv2.line(img, tuple(cube_corners_2d[0][0]), tuple(cube_corners_2d[4][0]), blue, line_width)
    cv2.line(img, tuple(cube_corners_2d[1][0]), tuple(cube_corners_2d[5][0]), blue, line_width)
    cv2.line(img, tuple(cube_corners_2d[2][0]), tuple(cube_corners_2d[6][0]), blue, line_width)
    cv2.line(img, tuple(cube_corners_2d[3][0]), tuple(cube_corners_2d[7][0]), blue, line_width)

    # Draw the top in green
    cv2.line(img, tuple(cube_corners_2d[4][0]), tuple(cube_corners_2d[5][0]), green, line_width)
    cv2.line(img, tuple(cube_corners_2d[5][0]), tuple(cube_corners_2d[6][0]), green, line_width)
    cv2.line(img, tuple(cube_corners_2d[6][0]), tuple(cube_corners_2d[7][0]), green, line_width)
    cv2.line(img, tuple(cube_corners_2d[7][0]), tuple(cube_corners_2d[4][0]), green, line_width)


def resize_to_fit_screen(frame, screen_width, screen_height):
    h, w = frame.shape[:2]
    scale = min(screen_width / w, screen_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h))


def main():
    # Load calibration data
    camera_matrix, dist_coefs = load_calibration_data("camera_calibration.json")

    # 3D cube definition (keep existing)
    _3d_corners = np.float32([
        [0, 0, 0], [0, 100, 0], [100, 100, 0], [100, 0, 0],
        [0, 0, -100], [0, 100, -100], [100, 100, -100], [100, 0, -100]
    ])

    # Camera initialization (modified to match working script)
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    if cam_list.GetSize() == 0:
        print("No cameras detected.")
        system.ReleaseInstance()
        return

    cam = cam_list.GetByIndex(0)
    cam.Init()

    # Start acquisition
    cam.BeginAcquisition()
    print("Streaming started. Press 'q' to quit.")

    # Get screen dimensions
    screen = screeninfo.get_monitors()[0]
    screen_width, screen_height = screen.width, screen.height

    
    while True:
        # Get next image with timeout
        image_result = cam.GetNextImage()  # 1 second timeout

        if image_result.IsIncomplete():
            print(f"Skipped incomplete image: {image_result.GetImageStatus()}")
            image_result.Release()
            continue

        # Convert to OpenCV format
        image_data = image_result.GetNDArray()
        frame = cv2.cvtColor(image_data, cv2.COLOR_BAYER_BG2BGR)
        
        # Chessboard processing (keep existing)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pattern_size = (10, 7)
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        print(found)
        # if found:
        #     # Refine corner positions
        #     term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        #     corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)

        #     # Solve the PnP problem to get rotation and translation vectors
        #     _, rvec, tvec = cv2.solvePnP(_3d_corners[:pattern_size[0] * pattern_size[1]], corners, camera_matrix, dist_coefs)

        #     # Project the 3D cube corners onto the 2D image plane
        #     cube_corners_2d, _ = cv2.projectPoints(_3d_corners, rvec, tvec, camera_matrix, dist_coefs)

        #     # Draw the cube on the frame
        #     draw_cube(frame, cube_corners_2d)

            # Display handling (modified)
        frame_resized = resize_to_fit_screen(frame, screen_width, screen_height)
        cv2.imshow("Real-Time Calibration Check", frame_resized)
        
        
        
        
        # Release image immediately after processing
        image_result.Release()

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

   
    # Cleanup (modified to match working script)
    cam.EndAcquisition()
    cam.DeInit()
    cam_list.Clear()
    system.ReleaseInstance()
    cv2.destroyAllWindows()
    print("Streaming stopped.")

# def main():
    # system = PySpin.System.GetInstance()
    # cam_list = system.GetCameras()
    # if cam_list.GetSize() == 0:
    #     print("No cameras detected.")
    #     system.ReleaseInstance()
    #     return
    # cam = cam_list.GetByIndex(0)
    # cam.Init()
    # cam.BeginAcquisition()
    # print("Streaming started. Press 'q' to exit.")
    # screen = screeninfo.get_monitors()[0]
    # screen_width, screen_height = screen.width, screen.height

    # while True:
    #     image = cam.GetNextImage()
    #     if image.IsIncomplete():
    #         print("Image incomplete with status: %d" % image.GetImageStatus())
    #         continue
    #     img_data = image.GetNDArray()
    #     img_bgr = cv2.cvtColor(img_data, cv2.COLOR_BAYER_BG2BGR)

    #     h, w = img_bgr.shape[:2]
    #     scale = min(screen_width / w, screen_height / h)
    #     new_w, new_h = int(w * scale), int(h * scale)
    #     img_resized = cv2.resize(img_bgr, (new_w, new_h))
    #     cv2.imshow("Camera Stream", img_resized)
    #     image.Release()
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cam.EndAcquisition()
    # cam.DeInit()
    # cam_list.Clear()
    # system.ReleaseInstance()
    # cv2.destroyAllWindows()
    # print("Streaming stopped.")


    

if __name__ == "__main__":
    main()
