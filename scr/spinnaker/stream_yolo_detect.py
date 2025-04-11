import PySpin
import cv2
import numpy as np
import screeninfo
from ultralytics import YOLO

# Load YOLO model (change path for YOLOv8 or YOLOv11)
model = YOLO("yolo11n.pt")  # Change to "yolov11n.pt" if using YOLOv11

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

    while True:
        image = cam.GetNextImage()

        if image.IsIncomplete():
            print("Image incomplete with status: %d" % image.GetImageStatus())
            continue

        # Convert Spinnaker image to OpenCV format
        img_data = image.GetNDArray()
        img_bgr = cv2.cvtColor(img_data, cv2.COLOR_BAYER_BG2BGR)

        # Run YOLO detection
        results = model(img_bgr)

        # Draw bounding boxes on the frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                conf = box.conf[0]  # Confidence score
                cls = int(box.cls[0])  # Class ID
                label = f"{model.names[cls]} {conf:.2f}"  # Class name and confidence

                # Draw bounding box and label
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Resize to fit screen while maintaining aspect ratio
        h, w = img_bgr.shape[:2]
        scale = min(screen_width / w, screen_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_bgr, (new_w, new_h))

        # Display the frame
        cv2.imshow("YOLO Object Detection", img_resized)

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
