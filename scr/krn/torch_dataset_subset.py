import numpy as np
import config
import json
import os
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader, Dataset
import config as config


class KeypointsSubsetDataset(Dataset):
    def __init__(self, img_folder, json_file, num_kpts, transform=None, target_size=(224, 224), split='train', num_train=21600, num_val=2700, keypoint_indices=None):
        """
        Args:
            img_folder (str): Path to the folder containing images.
            json_file (str): Path to the JSON file containing keypoint annotations.
            num_kpts (int): Total number of keypoints in the dataset.
            transform (callable, optional): Transform to be applied on the images and keypoints.
            target_size (tuple): Target size for the cropped and padded images.
            split (str): Dataset split ('train', 'val', or 'test').
            train_ratio (float): Ratio of the dataset to use for training.
            val_ratio (float): Ratio of the dataset to use for validation.
            keypoint_indices (list, optional): List of keypoint indices to include in the dataset. If None, all keypoints are used.
        """
        super().__init__()
        with open(json_file, "r") as jf:
            self.data = json.load(jf)
        self.num_kpts = num_kpts
        self.img_folder = img_folder
        self.image_names = sorted(os.listdir(img_folder), key=lambda x: int(x[4:-4]))
        self.image_names = [os.path.join(img_folder, filename) for filename in self.image_names]
        self.transform = transform
        self.target_size = target_size
        self.keypoint_indices = keypoint_indices if keypoint_indices is not None else list(range(num_kpts))

        num_images = len(self.image_names)
        train_end = num_train
        val_end = num_train + num_val

        if split == 'train':
            self.image_names = self.image_names[:train_end]
            self.data = self.data[:train_end]
        elif split == 'val':
            self.image_names = self.image_names[train_end:val_end]
            self.data = self.data[train_end:val_end]
        elif split == 'test':
            self.image_names = self.image_names[val_end:]
            self.data = self.data[val_end:]
        else:
            raise ValueError("Invalid split. Choose from 'train', 'val', or 'test'.")

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image = cv2.imread(str(self.image_names[index]), cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        labels = np.array(self.data[index]['keypoints'])
        labels = labels.reshape(self.num_kpts, 2)

        # Filter keypoints based on the user-defined indices
        labels = labels[self.keypoint_indices]

        bbox = np.array(self.data[index]['bbox_xyxy'])
        x_min, y_min, x_max, y_max = bbox

        cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]

        # Adjust keypoints to the cropped region
        labels[:, 0] -= x_min
        labels[:, 1] -= y_min

        # Pad the cropped image to make it square
        orig_h, orig_w = cropped_image.shape[:2]
        if orig_h > orig_w:
            pad = (orig_h - orig_w) // 2
            padded_image = cv2.copyMakeBorder(cropped_image, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            labels[:, 0] += pad
        else:
            pad = (orig_w - orig_h) // 2
            padded_image = cv2.copyMakeBorder(cropped_image, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            labels[:, 1] += pad

        # Scale keypoints to the target size
        scale_x = self.target_size[1] / padded_image.shape[1]
        scale_y = self.target_size[0] / padded_image.shape[0]
        labels[:, 0] *= scale_x
        labels[:, 1] *= scale_y

        padded_image = cv2.resize(padded_image, self.target_size)

        if self.transform:
            augmentations = self.transform(image=padded_image, keypoints=labels)
            padded_image = augmentations["image"]
            labels = augmentations["keypoints"]
        labels = np.array(labels).reshape(-1)

        return padded_image, labels.astype(np.float32)
    

if __name__ == "__main__":

    images_dir = config.IMG_DIR
    labels_path = config.LABELS_JSON
    num_kpts = config.NUM_KPTS

    # Example: Train on keypoints 0, 1, and 2
    keypoint_indices = [-4, -3, -2, -1]

    ds_train = KeypointsSubsetDataset(img_folder=images_dir,
                                      json_file=labels_path,
                                      num_kpts=num_kpts,
                                      transform=config.train_transforms,
                                      split='train',
                                      keypoint_indices=keypoint_indices)
    
    ds_val = KeypointsSubsetDataset(img_folder=images_dir,
                                    json_file=labels_path,
                                    num_kpts=num_kpts,
                                    transform=config.val_transforms,
                                    split='val',
                                    keypoint_indices=keypoint_indices)
    
    ds_test = KeypointsSubsetDataset(img_folder=images_dir,
                                     json_file=labels_path,
                                     num_kpts=num_kpts,
                                     transform=config.val_transforms,
                                     split='test',
                                     keypoint_indices=keypoint_indices)
                            
    train_loader = DataLoader(ds_train, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=1, shuffle=True, num_workers=0)
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=True, num_workers=0)

    for idx, (x, y) in enumerate(test_loader):
        plt.imshow(x[0][0].detach().cpu().numpy(), cmap='gray')  # [0][0] - for taking 1st example and 1st channel
        plt.plot(y[0][0::2].detach().cpu().numpy(), y[0][1::2].detach().cpu().numpy(), "go", markersize=4)
        plt.show()
