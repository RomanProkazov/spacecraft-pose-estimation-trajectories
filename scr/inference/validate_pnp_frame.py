import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from scr.krn import config as config
from scr.utils.utils_pnp import *


def do_pnp_frame(image_path, labels, sat_model, cmt):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    # Ground truth data
    q_gt = np.array(labels['pose'])
    t_gt = np.array(labels['translation'])
    image_points = np.array(labels['keypoints'])

    q_pr, t_pr = pnp(sat_model, image_points, cmt)

    t_error = error_translation(t_pr, t_gt)
    r_error = error_orientation(q_pr, q_gt)

    # print(f"Translation error (m): {t_error:.3f}")
    # print(f"Rotation error (deg): {r_error:.3f}")
    # print(f"Ground truth\nRotation:{q_gt}         Translation{t_gt}")
    # print(f"PnP\nRotation:{q_pr}         Translation{t_pr}")

    return t_error, r_error



if __name__ == "__main__":

    idx = 99
    camera_sat_model = config.SAT_CAM_JSON
    image_folder_path = Path(config.TEST_IMG_DIR)
    json_path = config.LABELS_JSON

    with open(config.SAT_CAM_JSON, 'r') as json_file:
        data = json.load(json_file)
    sat_model, cmt = np.array(data['sat_model']), np.array(data['camera_matrix'])
    distCoeffs_real = np.array(data['distortion_coefficients'])

    image_path_list = [image for image in sorted(image_folder_path.rglob('*.jpg'), key=lambda x: int(x.stem.split('_')[-1]))]
    with open(json_path, 'r') as f:
        annotations = json.load(f)
  
    do_pnp_frame(image_path_list[idx], annotations[idx], sat_model, cmt)
