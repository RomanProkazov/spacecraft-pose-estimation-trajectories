import PySpin
import cv2
import numpy as np
import screeninfo
from ultralytics import YOLO

# Load YOLO-Seg model (change to "yolov11-seg.pt" if using YOLOv11)
model = YOLO("yolo11n-seg.pt")

def main():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    if cam_list.GetSize() == 0:
        print("No cameras detected.")
        system.ReleaseInstance()
        return

    cam = cam_list.GetByIndex(0)
    cam.Init()
    cam.BeginAcquisition()
    print("Streaming started. Press 'q' to exit.")

    # Get screen size for proper resizing
    screen = screeninfo.get_monitors()[0]
    screen_width, screen_height = screen.width, screen.height

    try:
        while True:
            image = cam.GetNextImage()

            if image.IsIncomplete():
                print("Image incomplete with status: %d" % image.GetImageStatus())
                continue

            # Convert Spinnaker image to OpenCV format
            img_data = image.GetNDArray()
            img_bgr = cv2.cvtColor(img_data, cv2.COLOR_BAYER_BG2BGR)

            # Run YOLO segmentation
            results = model(img_bgr)

            # Process each detected object
            for result in results:
                # Check if there are any detections
                if result.masks is not None:
                    for seg, box in zip(result.masks.xy, result.boxes):
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                        conf = box.conf[0]  # Confidence score
                        cls = int(box.cls[0])  # Class ID
                        label = f"{model.names[cls]} {conf:.2f}"  # Class name and confidence
                        
                        # Create mask and overlay on image
                        mask = np.zeros_like(img_bgr[:, :, 0], dtype=np.uint8)
                        cv2.fillPoly(mask, [np.array(seg, dtype=np.int32)], 255)
                        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                        mask_colored[:, :, 1:] = 0  # Set color to blue
                        img_bgr = cv2.addWeighted(img_bgr, 1, mask_colored, 0.5, 0)

                        # Draw bounding box and label
                        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Resize to fit screen while maintaining aspect ratio
            h, w = img_bgr.shape[:2]
            scale = min(screen_width / w, screen_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img_bgr, (new_w, new_h))

            # Display the frame
            cv2.imshow("YOLO Segmentation", img_resized)

            # Release image buffer
            image.Release()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Cleanup
        cam.EndAcquisition()
        cam.DeInit()
        cam_list.Clear()
        system.ReleaseInstance()
        cv2.destroyAllWindows()
        print("Streaming stopped.")

if __name__ == "__main__":
    main()