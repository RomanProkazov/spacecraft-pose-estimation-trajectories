import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from scr.krn import config as config
from scr.utils.utils_pnp import *
from validate_pnp_frame import do_pnp_frame

def validate_pnp_loop(image_folder_path,
                    json_data_path,
                    camera_sat_json,
                    ):                           
    with open(json_data_path, 'r') as f:
        annotations = json.load(f)

    image_path_list = sorted([image for image in Path(image_folder_path).rglob('*.jpg')], key=lambda x: int(x.stem.split('_')[-1]))

    with open(camera_sat_json, 'r') as json_file:
        data = json.load(json_file)
    sat_model, cmt = np.array(data['sat_model']), np.array(data['camera_matrix']) 

    t_error_sum, r_error_sum = 0, 0
    num_images = len(image_path_list)
    for idx in range(num_images):
        t_error, r_error = do_pnp_frame(image_path_list[idx], annotations[idx], sat_model, cmt)
        t_error_sum += t_error
        r_error_sum += r_error

    print(f"Average translation error (m): {t_error_sum/num_images:.3f}")
    print(f"Average rotation error (deg): {r_error_sum/num_images:.3f}")


if __name__ == "__main__":
    camera_sat_model = config.SAT_CAM_JSON
    image_folder_path = Path(config.TEST_IMG_DIR)
    json_path = config.LABELS_JSON
    validate_pnp_loop(image_folder_path, json_path, camera_sat_model)


