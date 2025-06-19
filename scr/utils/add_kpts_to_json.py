from tqdm import tqdm
import numpy as np
import json
from utils_pnp import *
import sys
sys.path.append("../../scr/krn")
import config as config
from general_utils import(load_camera_matrix_sat_model,
                                    load_images,
                                    load_labels)



def add_kpts_to_json(image_folder_path,
                     json_data_path,
                     camera_sat_json,
                     output_json,
                     res=(120, 720), unit_scale=1):

    image_path_list = load_images(image_folder_path)
    annotations = load_labels(json_data_path)
    cmt, sat_model = load_camera_matrix_sat_model(camera_sat_json)

    # Clean data from blender unused info
    keys_to_remove = ["position", "distance", "background", "lighting"] # "offset"
    annotations = [
        {key: value for key, value in element.items() if key not in keys_to_remove}
        for element in annotations
    ]
    
    num_images = len(image_path_list)
    for idx in tqdm(range(num_images),
                    total=num_images,
                    desc="Images processed",
                    ncols=80):
        
        labels = annotations[idx] 

        # Ground truth data
        q_gt =np.array(labels['pose'])
        t_gt = np.array(labels['translation'])*unit_scale

        distCoeffs = np.zeros((5, 1), dtype=np.float32)
        image_points = project_keypoints(q_gt, t_gt, cmt, distCoeffs, sat_model)
        image_points = image_points.T
        image_points_marker = image_points[-4:]

        # Calculate bbox coords from keypoints    
        x_min = np.min(image_points[:, 0])
        y_min = np.min(image_points[:, 1])
        x_max = np.max(image_points[:, 0])
        y_max = np.max(image_points[:, 1])

        # Clip bounding box coordinates to the frame
        x_min = max(0, min(x_min, res[0]))
        y_min = max(0, min(y_min, res[1]))
        x_max = max(0, min(x_max, res[0]))
        y_max = max(0, min(y_max, res[1]))

        # Marker bounding box  
        x_min_marker = np.min(image_points_marker[:, 0])
        y_min_marker = np.min(image_points_marker[:, 1])
        x_max_marker = np.max(image_points_marker[:, 0])
        y_max_marker = np.max(image_points_marker[:, 1])

        bbox = [x_min, y_min, x_max, y_max]
        bbox_marker = [x_min_marker, y_min_marker, x_max_marker, y_max_marker]

        labels['keypoints'] = image_points.tolist()
        labels['filename'] = image_path_list[idx].name
        labels['translation'] = t_gt.tolist()
        labels['bbox_xyxy'] = bbox
        labels['bbox_marker'] = bbox_marker

    with open(output_json, "w") as json_file:
        json.dump(annotations, json_file, indent=4)


if __name__ == "__main__":
    output_json = config.LABELS_JSON
    add_kpts_to_json(image_folder_path=config.IMG_DIR,
                     json_data_path=config.LABELS_BLENDER_JSON,
                     camera_sat_json=config.SAT_CAM_JSON,
                     output_json=output_json,
                     res=(1280, 720), unit_scale=1)