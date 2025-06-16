import os
import shutil
from pathlib import Path

def copy_last_300_images_per_1000(src_folder, dst_folder):
    src_path = Path(src_folder)
    dst_path = Path(dst_folder)
    dst_path.mkdir(parents=True, exist_ok=True)

    # List all image files (common image extensions)
    images = sorted([f for f in src_path.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']],
                    key=lambda x: int(x.stem.split('_')[-1]) if x.stem.split('_')[-1].isdigit() else -1)

    # Process in chunks of 1000
    chunk_size = 1000
    keep_last = 300

    for i in range(0, len(images), chunk_size):
        chunk = images[i:i+chunk_size]
        # Take last 300 images from the chunk
        last_300 = chunk[-keep_last:]
        for img_path in last_300:
            shutil.copy(img_path, dst_path / img_path.name)

    print(f"Copied last {keep_last} images from each {chunk_size} images chunk from {src_folder} to {dst_folder}.")

# Example usage:
# copy_last_300_images_per_1000('/path/to/source', '/path/to/destination')
if __name__ == "__main__":
    src_folder = '../../data/images'  # Source folder containing images
    dst_folder = '../../data/images_last_300_marker'  # Destination folder to copy images
    copy_last_300_images_per_1000(src_folder, dst_folder)