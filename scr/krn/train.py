import torch
from torch_dataset import KeypointsDataset
from torch_dataset_subset import KeypointsSubsetDataset
from torch import nn, optim
import config
from tqdm import tqdm
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from torch_utils import save_checkpoint


def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    losses = []
    loop = tqdm(loader, ncols=80)
    num_examples = 0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = loss_fn(scores, targets)
        num_examples += torch.numel(scores)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss average over epoch: {(sum(losses) / num_examples) ** 0.5}")


def validate(loader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    num_examples = 0
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = loss_fn(scores, targets)
            total_loss += loss.item()
            num_examples += torch.numel(scores)

    avg_loss = (total_loss / num_examples) ** 0.5
    print(f"Validation RMSE: {avg_loss}")
    return avg_loss


def do_training(image_folder, labels_path, keypoint_indices):
    num_kpts = len(keypoint_indices)  # Set num_kpts based on the number of keypoints being trained on

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

    val_ds = KeypointsSubsetDataset(
        img_folder=image_folder,
        json_file=labels_path,
        num_kpts=config.NUM_KPTS,  # Total number of keypoints in the dataset
        transform=config.val_transforms,
        split="val",
        keypoint_indices=keypoint_indices,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, shuffle=False
    )

    loss_fn = nn.MSELoss(reduction="sum")
    model = EfficientNet.from_pretrained("efficientnet-b0")
    model._fc = nn.Linear(1280, num_kpts * 2)  # Adjust the output size based on the number of keypoints
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")
    best_checkpoint = None

    for i, epoch in enumerate(range(config.NUM_EPOCHS)):
        print(f"Epoch {i + 1}")
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)

        # Validate and save the best model
        val_loss = validate(val_loader, model, loss_fn, config.DEVICE)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            print(f"New best model found with validation RMSE: {best_val_loss}")
            save_checkpoint(best_checkpoint, filename=config.CHECKPOINT_FILE)


if __name__ == "__main__":
    keypoint_indices = [-4, -3, -2, -1]  # Train on keypoints 0, 1, 2, and 3
    do_training(
        image_folder=config.IMG_DIR,
        labels_path=config.LABELS_JSON,
        keypoint_indices=keypoint_indices,
    )