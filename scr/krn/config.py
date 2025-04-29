import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torchvision.transforms as transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3  # Reduced learning rate
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 128  # Single batch for overfitting
NUM_EPOCHS = 100
NUM_WORKERS = 16
PIN_MEMORY = True
SAVE_MODEL = True
LOAD_MODEL = True

NUM_KPTS = 16
NUM_KPTS_INF = 4
CHECKPOINT_FILE = "../../trained_models/krn-dir/b0_4kpts_3072px_v1.pth.tar"
KRN_MODEL_PATH = "../../trained_models/krn-dir/b0_4kpts_3072px_v1.pth.tar"
ODN_MODEL_PATH = "../../trained_models/odn-dir/best_odn_3072px_v1.pt"
SEG_MODEL_PATH = "../../runs/segment/train5/weights/best.pt"

IMG_DIR = "../../data_3072px/images"
TEST_IMG_DIR = "../../data_splitted/data_odn/test/images"

SAT_CAM_JSON = "../../data_3072px/labels/cam_sat.json"
LABELS_JSON = "../../data_3072px/labels/labels_sat_27kimgs.json"

INPUT_VIDEO_PATH = "../../videos/3072_2048/test/trajectories_videos/trajectory_1.mp4"
OUTPUT_VIDEO_PATH_SEG = "../../videos/3072_2048/test/seg_videos/seg_trajectory1.mp4"



# Data augmentation for images
train_transforms = A.Compose(
    [
        A.Resize(224, 224),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.8),
        A.RandomBrightnessContrast(contrast_limit=0.5, brightness_limit=0.5, p=0.2),
        A.OneOf([
            A.GaussNoise(p=0.8),
            A.CLAHE(p=0.8),
            A.ImageCompression(p=0.8),
            A.RandomGamma(p=0.8),
            A.Posterize(p=0.8),
            A.Blur(p=0.8),
        ], p=1.0),
        A.OneOf([
            A.GaussNoise(p=0.8),
            A.CLAHE(p=0.8),
            A.ImageCompression(p=0.8),
            A.RandomGamma(p=0.8),
            A.Posterize(p=0.8),
            A.Blur(p=0.8),
        ], p=1.0),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.2, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(
            mean=[-0.6018, -0.6015, -0.6016],
            std=[0.5889, 0.5888, 0.5888],
            max_pixel_value=255.0,
        ),
        
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
)
val_transforms = A.Compose(
    [   A.Resize(224, 224),
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
)
test_transforms = A.Compose(
    [
        # A.LongestMaxSize(max_size=640),

        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ]
)
mean_std_transforms = A.Compose(
    [   A.Resize(224, 224),
        ToTensorV2(),
    ] #, keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
)