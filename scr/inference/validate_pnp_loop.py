import sys
import json
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from scr.krn import config as config
from scr.utils.utils_pnp import *
from validate_pnp_frame import do_pnp_frame
from scr.utils.general_utils import load_images, load_labels, load_camera_matrix_sat_model


def validate_pnp_loop(image_folder_path,
                    json_data_path,
                    camera_sat_json,
                    start_idx=0):
                               
    labels = load_labels(json_data_path)
    image_path_list = load_images(image_folder_path)
    image_path_list = image_path_list[start_idx:]

    cmt, sat_model = load_camera_matrix_sat_model(camera_sat_json)

    t_error_sum, r_error_sum = 0, 0
    num_images = len(image_path_list)
    for idx in range(num_images):
        t_error, r_error = do_pnp_frame(labels[idx], sat_model, cmt)
        t_error_sum += t_error
        r_error_sum += r_error

    print(f"Average translation error (m): {t_error_sum/num_images:.3f}")
    print(f"Average rotation error (deg): {r_error_sum/num_images:.3f}")


def main():
    camera_sat_model = config.SAT_CAM_JSON
    image_folder_path = Path(config.IMG_DIR)
    json_path = config.LABELS_JSON
    validate_pnp_loop(image_folder_path, json_path, camera_sat_model, start_idx=0000)


if __name__ == "__main__":
    main()

