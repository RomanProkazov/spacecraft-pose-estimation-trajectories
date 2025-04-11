import PySpin
import cv2
import numpy as np
import screeninfo

def main():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    if cam_list.GetSize() == 0:
        print("No cameras detected.")
        system.ReleaseInstance()
        return

    cam = cam_list.GetByIndex(0)
    cam.Init()


    #     # Disable autofocus if supported
    # try:
    #     node_map = cam.GetNodeMap()
    #     focus_auto = PySpin.CEnumerationPtr(node_map.GetNode("FocusAuto"))
    #     if PySpin.IsAvailable(focus_auto) and PySpin.IsWritable(focus_auto):
    #         focus_auto_off = PySpin.CEnumEntryPtr(focus_auto.GetEntryByName("Off"))
    #         if PySpin.IsAvailable(focus_auto_off) and PySpin.IsReadable(focus_auto_off):
    #             focus_auto.SetIntValue(focus_auto_off.GetValue())
    #             print("Auto focus disabled.")

    #     # Optional: set manual focus position (value depends on your lens)
    #     focus_pos = PySpin.CFloatPtr(node_map.GetNode("FocusPosition"))
    #     if PySpin.IsAvailable(focus_pos) and PySpin.IsWritable(focus_pos):
    #         focus_pos.SetValue(15.0)  # adjust this value as needed
    #         print("Focus position set to 15.0")

    # except Exception as e:
    #     print(f"Could not disable autofocus: {e}")


    cam.BeginAcquisition()
    print("Streaming started. Press 'q' to exit.")

    # Get screen size for proper resizing
    screen = screeninfo.get_monitors()[0]
    screen_width = screen.width
    screen_height = screen.height

    while True:
        image = cam.GetNextImage()

        if image.IsIncomplete():
            print("Image incomplete with status: %d" % image.GetImageStatus())
            continue

        # Convert image to OpenCV format
        img_data = image.GetNDArray()
        img_bgr = cv2.cvtColor(img_data, cv2.COLOR_BAYER_BG2BGR)

        # Resize to fit screen while maintaining aspect ratio
        h, w = img_bgr.shape[:2]
        scale = min(screen_width / w, screen_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_bgr, (new_w, new_h))

        # Display the frame
        cv2.imshow("Camera Stream", img_resized)

        # Release image buffer
        image.Release()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.EndAcquisition()
    cam.DeInit()
    cam_list.Clear()
    system.ReleaseInstance()
    cv2.destroyAllWindows()
    print("Streaming stopped.")

if __name__ == "__main__":
    main()
