import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image


def segment_frame(image, model):
    results = model(image)
    masks = results[0].masks.data.cpu().numpy()  # Segmentation masks
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    classes = results[0].boxes.cls.cpu().numpy()  # Class labels
    return results


def visualize_segmentation(image, masks, boxes, classes, class_names):
    overlay = image.copy()

    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(masks))]

    for i, mask in enumerate(masks):
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_resized = mask_resized.astype(bool)

        overlay[mask_resized] = cv2.addWeighted(overlay[mask_resized], 0.5, np.array(colors[i]), 0.5, 0)

        # Draw bounding box
        x_min, y_min, x_max, y_max = map(int, boxes[i])
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), colors[i], 2)

        # Add class label
        label = f"{class_names[int(classes[i])]}"
        cv2.putText(overlay, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

    # Display the image
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Segmentation Visualization")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Load YOLO segmentation model
    model = YOLO("../../runs/segment/train5/weights/best.pt")  

    # Load an input image
    image_path = "../../data_splitted/data_seg/test/images/img_26326.jpg"  
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        exit()


    # Perform segmentation
    results = segment_frame(image, model)

    # Show the results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        # r.save_crop(save_dir, file_name=Path(random_image.name))
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        plt.figure(figsize=(12, 7))
        plt.imshow(im)
        plt.axis(False)
        plt.show()

    # # Visualize segmentation
    # class_names = model.names  # Class names from the YOLO model
    # visualize_segmentation(image, masks, boxes, classes, class_names)
