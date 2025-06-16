from pathlib import Path
import json
import sys
sys.path.append("../../scr/krn")
import config as config
 
def kpts_txt_from_json_keypoints(path_to_json_file, labels_path_bbox, res, kpts_start_idx=None, kpts_end_idx=None):
 
    labels_path_bbox = Path(labels_path_bbox)
    labels_path_bbox.mkdir(parents=True, exist_ok=True)
    
    with open(path_to_json_file, 'r') as file:
        data = json.load(file)
  
    for sample_data in data:
        kpts = sample_data['keypoints']
        kpts = kpts[kpts_start_idx:kpts_end_idx]
        # xmin, ymin, xmax, ymax = sample_data['bbox_xyxy']
        xmin, ymin, xmax, ymax = sample_data['bbox_marker']
        width  = xmax - xmin
        height = ymax - ymin
        x_c    = (xmax + xmin) / 2
        y_c    = (ymax + ymin) / 2
 
        # Normalize coordinates
        x_c /= res[0]
        y_c /= res[1]
        width /= res[0]
        height /= res[1]
 
        bbox_coords = x_c, y_c, width, height
        
        image_name = sample_data['filename']
        sample = image_name[:-4]
        with open(labels_path_bbox / f"{sample}.txt", 'w') as f:
            f.write(f"0 {bbox_coords[0]:.6f} {bbox_coords[1]:.6f} {bbox_coords[2]:.6f} {bbox_coords[3]:.6f} ")
 
            for kpt in kpts:
                u, v = kpt
                u, v = u/res[0], v/res[1]
                f.write(f"{round(u, 2)} {round(v, 2)} ")
 
 
if __name__ == "__main__":
    path_to_json_file = config.LABELS_JSON
    labels_path_bbox = "../../data/labels/kpts_yolo_marker"
    res = (1280, 720)
    kpts_txt_from_json_keypoints(path_to_json_file, labels_path_bbox, res, kpts_start_idx=-4, kpts_end_idx=None)
    print("Done!")
 