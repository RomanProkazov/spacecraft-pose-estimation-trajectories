import PySpin
import cv2
import numpy as np
import time
import os
import screeninfo

def main():
    # Create output directory
    output_dir = "calibration_images"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize camera
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    if cam_list.GetSize() == 0:
        print("No camera detected.")
        cam_list.Clear()
        system.ReleaseInstance()
        return

    try:
        cam = cam_list.GetByIndex(0)
        cam.Init()
        
        # First check available pixel formats
        node_map = cam.GetNodeMap()
        pixel_format = PySpin.CEnumerationPtr(node_map.GetNode("PixelFormat"))
        
        if not PySpin.IsAvailable(pixel_format) or not PySpin.IsReadable(pixel_format):
            print("Pixel format node not available")
            return

        # Print available pixel formats
        print("Available pixel formats:")
        entries = pixel_format.GetEntries()
        for entry in entries:
            if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
                print(f"- {entry.GetSymbolic()}")

        # Try to set BayerBG8 format if available
        bayer_entry = pixel_format.GetEntryByName("BayerBG8")
        if PySpin.IsAvailable(bayer_entry) and PySpin.IsReadable(bayer_entry):
            try:
                pixel_format.SetIntValue(bayer_entry.GetValue())
                print("Set pixel format to BayerBG8")
            except PySpin.SpinnakerException as ex:
                print(f"Could not set BayerBG8 format: {ex}")
                print("Using current pixel format instead")
        else:
            print("BayerBG8 format not available, using current format")

        # Set acquisition mode to continuous
        acquisition_mode = PySpin.CEnumerationPtr(node_map.GetNode("AcquisitionMode"))
        acquisition_mode_continuous = acquisition_mode.GetEntryByName("Continuous")
        acquisition_mode.SetIntValue(acquisition_mode_continuous.GetValue())

        # Get screen dimensions
        screen = screeninfo.get_monitors()[0]
        screen_width, screen_height = screen.width, screen.height

        # Start acquisition
        cam.BeginAcquisition()
        print("Streaming started. Press:")
        print("  's' - Save current frame")
        print("  'q' - Quit")

        while True:
            try:
                # Capture frame with timeout
                image_result = cam.GetNextImage(1000)
                
                if image_result.IsIncomplete():
                    print(f"Skipped incomplete image: {image_result.GetImageStatus()}")
                    image_result.Release()
                    continue

                # Convert to OpenCV format
                image_data = image_result.GetNDArray()
                
                # Handle different pixel formats
                current_format = pixel_format.GetCurrentEntry().GetSymbolic()
                if "Bayer" in current_format:
                    # Bayer pattern needs conversion
                    if "BG" in current_format:
                        frame = cv2.cvtColor(image_data, cv2.COLOR_BayerBG2BGR)
                    elif "RG" in current_format:
                        frame = cv2.cvtColor(image_data, cv2.COLOR_BayerRG2BGR)
                    elif "GB" in current_format:
                        frame = cv2.cvtColor(image_data, cv2.COLOR_BayerGB2BGR)
                    elif "GR" in current_format:
                        frame = cv2.cvtColor(image_data, cv2.COLOR_BayerGR2BGR)
                    else:
                        frame = cv2.cvtColor(image_data, cv2.COLOR_BayerBG2BGR)  # Default
                else:
                    # Mono or other format
                    frame = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
                
                # Resize for display
                h, w = frame.shape[:2]
                scale = min(screen_width / w, screen_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                frame_resized = cv2.resize(frame, (new_w, new_h))

                # Display frame
                cv2.imshow("Camera Stream (Press 's' to save)", frame_resized)
                
                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    timestamp = int(time.time() * 1000)
                    filename = os.path.join(output_dir, f"calib_{timestamp}.png")
                    cv2.imwrite(filename, frame)
                    print(f"Saved: {filename}")
                    # Visual feedback
                    cv2.putText(frame_resized, "SAVED!", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Camera Stream (Press 's' to save)", frame_resized)
                    cv2.waitKey(300)  # Show "SAVED" message for 300ms
                    
                elif key == ord('q'):
                    break

                # Release image buffer
                image_result.Release()

            except PySpin.SpinnakerException as ex:
                print(f"Error during acquisition: {ex}")
                break

    except Exception as ex:
        print(f"Error: {ex}")
    finally:
        # Cleanup
        try:
            if 'cam' in locals() and cam.IsInitialized():
                cam.EndAcquisition()
                cam.DeInit()
            cam = None  # Explicitly delete the camera reference
        except Exception as ex:
            print(f"Error during camera cleanup: {ex}")
        finally:
            cam_list.Clear()
            system.ReleaseInstance()
            cv2.destroyAllWindows()
            print("Streaming stopped.")
if __name__ == "__main__":
    main()