import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from scr.krn import config as config
from scr.utils.utils_pnp import *
from validate_pnp_frame import do_pnp_frame


def load_labels(json_data_path):
    with open(json_data_path, 'r') as f:
        labels = json.load(f)
    return labels


def load_images(image):
    image_path_list = sorted([image for image in Path(image).rglob('*.jpg')],
                             key=lambda x: int(x.stem.split('_')[-1]))
    return image_path_list


def validate_pnp_loop(image_folder_path,
                    json_data_path,
                    camera_sat_json,
                    start_idx):
                               
    labels = load_labels(json_data_path)
    image_path_list = load_images(image_folder_path)

    image_path_list = image_path_list[start_idx:]

    with open(camera_sat_json, 'r') as json_file:
        data = json.load(json_file)
    sat_model, cmt = np.array(data['sat_model']), np.array(data['camera_matrix']) 

    t_error_sum, r_error_sum = 0, 0
    num_images = len(image_path_list)
    for idx in range(num_images):
        t_error, r_error = do_pnp_frame(image_path_list[idx], labels[idx], sat_model, cmt)
        t_error_sum += t_error
        r_error_sum += r_error

    print(f"Average translation error (m): {t_error_sum/num_images:.3f}")
    print(f"Average rotation error (deg): {r_error_sum/num_images:.3f}")


def main():
    camera_sat_model = config.SAT_CAM_JSON
    image_folder_path = Path(config.TEST_IMG_DIR)
    json_path = config.LABELS_JSON_PREDS
    validate_pnp_loop(image_folder_path, json_path, camera_sat_model, start_idx=4500)


if __name__ == "__main__":
    main()

