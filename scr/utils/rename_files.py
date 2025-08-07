import os

# Change this to the folder where your images are stored
folder_path = "../../data_real/lux_sat_data_real_v4"

# List of valid image file extensions
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# Get the list of files in the directory
files = os.listdir(folder_path)

# Filter to only image files
images = [f for f in files if os.path.splitext(f)[1].lower() in valid_extensions]
images.sort()  # Sort for consistent renaming

# Rename images
for i, filename in enumerate(images, start=1):
    ext = os.path.splitext(filename)[1].lower()
    new_name = f"frame_bright_{i:04d}{ext}"
    src = os.path.join(folder_path, filename)
    dst = os.path.join(folder_path, new_name)
    if src != dst:
        os.rename(src, dst)
        print(f"Renamed '{filename}' -> '{new_name}'")