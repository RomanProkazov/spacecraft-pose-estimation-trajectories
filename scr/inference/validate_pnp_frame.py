import sys
from pathlib import Path
import numpy as np
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from scr.krn import config as config
from scr.utils.utils_pnp import do_ransac_lm, error_orientation, error_translation
from scr.utils.general_utils import(load_camera_matrix_sat_model,
                                    load_images,
                                    load_labels)


def do_pnp_frame(labels, sat_model, cmt):
    
    # Ground truth data
    q_gt = np.array(labels['pose'])
    t_gt = np.array(labels['translation'])
    image_points = np.array(labels['keypoints'])

    # q_pr, t_pr = pnp(sat_model, image_points, cmt)
    q_pr, t_pr = do_ransac_lm(sat_model, image_points, cmt)

    t_error = error_translation(t_pr, t_gt)
    r_error = error_orientation(q_pr, q_gt)

    # print(f"Translation error (m): {t_error:.3f}")
    # print(f"Rotation error (deg): {r_error:.3f}")
    # print(f"Ground truth\nRotation:{q_gt}         Translation{t_gt}")
    # print(f"PnP\nRotation:{q_pr}         Translation{t_pr}")

    return t_error, r_error


def main():
    idx = 1
    camera_sat_model = config.SAT_CAM_JSON
    image_folder_path = Path(config.IMG_DIR)
    json_path = config.LABELS_JSON

    image_path_list = load_images(image_folder_path)
    annotations = load_labels(json_path)
    cmt, sat_model = load_camera_matrix_sat_model(camera_sat_model)
  
    do_pnp_frame(annotations[idx], sat_model, cmt)


if __name__ == "__main__":
    main()

