import cv2
import os

def create_video_from_images(image_folder, output_video, fps=30, start_idx=0, end_idx=500):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")] 
    images.sort()  # Sort images by name

    if not images:
        print("No images found in the specified folder.")
        return

    frame = cv2.imread(os.path.join(image_folder, images[start_idx]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))  

    for image in images[start_idx:end_idx]:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()
    print(f"Video saved to {output_video}")

# Example usage:
image_folder = "../../data/images/trajectories_images"  
total_images = len([img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")])
frames_per_video = 500

for i in range(0, total_images, frames_per_video):
    start_idx = i
    end_idx = min(i + frames_per_video, total_images)
    output_video = f"../../data/images/trajectories_videos/trajectory_{i // frames_per_video + 1}.mp4"
    create_video_from_images(image_folder, output_video, fps=30, start_idx=start_idx, end_idx=end_idx)