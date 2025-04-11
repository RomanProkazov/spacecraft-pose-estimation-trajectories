import PySpin
import cv2
import numpy as np
import time
import screeninfo

def main():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    if cam_list.GetSize() == 0:
        print("No camera detected.")
        cam_list.Clear()
        system.ReleaseInstance()
        return

    cam = cam_list.GetByIndex(0)
    cam.Init()
    
    # Configure pixel format to BayerBG8 (color)
    node_map = cam.GetNodeMap()
    pixel_format = PySpin.CEnumerationPtr(node_map.GetNode("PixelFormat"))
    if PySpin.IsAvailable(pixel_format) and PySpin.IsWritable(pixel_format):
        pixel_format_entry = PySpin.CEnumEntryPtr(pixel_format.GetEntryByName("BayerBG8"))
        if PySpin.IsAvailable(pixel_format_entry) and PySpin.IsReadable(pixel_format_entry):
            pixel_format.SetIntValue(pixel_format_entry.GetValue())

    # time.sleep(3)  # Wait for 3 seconds
    
    # Get screen size for proper resizing
    screen = screeninfo.get_monitors()[0]
    screen_width = screen.width
    screen_height = screen.height

    cam.BeginAcquisition()

    while True:
        image_result = cam.GetNextImage()

        if image_result.IsIncomplete():
            print(f"Image incomplete with status {image_result.GetImageStatus()}")
        else:
            # Convert to BGR
            image_data = image_result.GetNDArray()
            image_bgr = cv2.cvtColor(image_data, cv2.COLOR_BayerBG2BGR)
            
            # Resize to fit screen while maintaining aspect ratio
            h, w = image_bgr.shape[:2]
            scale = min(screen_width / w, screen_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(image_bgr, (new_w, new_h))

            # Display the frame
            
            cv2.imshow("Camera Stream", img_resized)

          # Display for 2 seconds
            # cv2.imwrite("calibration_images2/img_2.png", image_bgr)
            # print("Image captured and saved as 'calibration_image.png'")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # time.sleep(5)
                timestamp = int(time.time() * 1000)
                
                cv2.imwrite(f"calibration_images5/img_{timestamp}.png", image_bgr)

    image_result.Release()
    cam.EndAcquisition()
    cam.DeInit()
    del cam
    cam_list.Clear()
    system.ReleaseInstance()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()