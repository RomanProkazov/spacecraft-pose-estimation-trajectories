import cv2
import os
import random
import matplotlib.pyplot as plt

def visualize_bbox(image_folder, label_folder):
    # Get list of all images and labels
    image_files = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    label_files = [lbl for lbl in os.listdir(label_folder) if lbl.endswith(".txt")]

    # Randomly select an image
    selected_image = random.choice(image_files)
    selected_label = selected_image.replace(".png", ".txt").replace(".jpg", ".txt")

    # Read the image
    image_path = os.path.join(image_folder, selected_image)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Read the corresponding label
    label_path = os.path.join(label_folder, selected_label)
    with open(label_path, 'r') as f:
        label_data = f.readline().strip().split()
        _, x_center, y_center, width, height = map(float, label_data)

    # Convert normalized coordinates to pixel values
    img_height, img_width, _ = image.shape
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    # Calculate bounding box coordinates
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)

    # Draw the bounding box on the image
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Display the image with the bounding box
    plt.imshow(image)
    plt.title(f"Image: {selected_image}")
    plt.show()

# Example usage:
image_folder = "../../data_splitted/val/images"
label_folder = "../../data_splitted/val/labels"
visualize_bbox(image_folder, label_folder)
