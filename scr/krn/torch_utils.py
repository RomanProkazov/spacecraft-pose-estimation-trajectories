import torch
import numpy as np
import config
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torch_dataset_subset import KeypointsSubsetDataset


def get_mean_std(loader):
    # VAR[X] = E[X**2] - E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches +=1

    mean = channels_sum/num_batches 
    std = (channels_squared_sum/num_batches - mean**2)**0.5

    return mean, std



def get_rmse(loader, model, loss_fn, device):
    model.eval()
    num_examples = 0
    losses = []
    with torch.inference_mode():
        for batch_idx, (data, targets) in enumerate(loader):
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = loss_fn(scores[targets != -1], targets[targets != -1]) # We compute the loss function wrt those ones that do not have -1
            num_examples += scores[targets != -1].shape[0]
            losses.append(loss.item())

    model.train()
    print(f"Loss on val: {(sum(losses)/num_examples)**0.5}")


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer=None, lr=None):
    print("=> Loading chekpoint")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if lr:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


if __name__ == "__main__":
    image_folder = "../../data/images/trajectories_images"
    labels_path = "../../data/labels/labels_5kimgs.json"
    keypoint_indices = [0, 1, 2, 3]

    train_ds = KeypointsSubsetDataset(
    img_folder=image_folder,
    json_file=labels_path,
    num_kpts=config.NUM_KPTS,  # Total number of keypoints in the dataset
    transform=config.train_transforms,
    split="train",
    keypoint_indices=keypoint_indices,
)
train_loader = DataLoader(
    train_ds, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, shuffle=True
)

mean, std = get_mean_std(train_loader)
print(mean)
print(std)