import numpy as np
import json
from pathlib import Path


def blender_to_opencv_coord(blender_points):
    """
    Converts 3D points from Blender's coordinate system to OpenCV's.
    Blender: +X right, +Y up, +Z forward
    OpenCV: +X right, +Y down, +Z forward (camera facing +Z)
    """
    blender_points = np.array(blender_points, dtype=np.float32) 
    opencv_points = blender_points.copy()
    opencv_points[:, 1] *= -1  # invert Y
    opencv_points[:, 2] *= -1  # invert Z
    return opencv_points


def load_camera_matrix_sat_model(camera_sat_json, object='sat_model'):
    with open(camera_sat_json, 'r') as json_file:
        data = json.load(json_file)
    sat_model, camera_matrix = np.array(data[object]), np.array(data['camera_matrix']) 
    sat_model_opencv = blender_to_opencv_coord(sat_model)
    return camera_matrix, sat_model_opencv


def load_labels(json_data_path):
    with open(json_data_path, 'r') as f:
        labels = json.load(f)
    return labels


def load_images(image_dir_path):
    image_path_list = sorted([image for image in Path(image_dir_path).rglob('*.jpg')],
                             key=lambda x: int(x.stem.split('_')[-1]))
    return image_path_list