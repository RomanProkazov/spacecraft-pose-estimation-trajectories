import torch
from torch_dataset import KeypointsDataset
from torch import nn, optim
import config
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet


def train_one_epoch(data, targets, model, optimizer, loss_fn, device, scheduler=None):
    losses = []
    num_examples = 0
    data = data.to(device=device)
    targets = targets.to(device=device)

    # forward
    scores = model(data)
    loss = loss_fn(scores, targets)
    print(f"Loss: {loss:.2f}")
    num_examples += torch.numel(scores)
    losses.append(loss.item())

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    mean_loss = (sum(losses) / num_examples) ** 0.5
    if scheduler:
        scheduler.step(mean_loss)

    print(f"Loss average over epoch: {mean_loss}")


def train_overfit(train_folder, json_file, num_kpts):
    train_ds = KeypointsDataset(
        img_folder=train_folder,
        json_file=json_file,
        num_kpts=num_kpts, transform=config.train_transforms,
        split="train"
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, shuffle=True
    )

    model = EfficientNet.from_pretrained("efficientnet-b0")
    model._fc = nn.Linear(1280, num_kpts * 2)
    model = model.to(config.DEVICE)

    loss_fn = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)  # weight_decay=config.WEIGHT_DECAY
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=100, verbose=True)

    data, targets = next(iter(train_loader))
    for i, epoch in enumerate(range(config.NUM_EPOCHS)):
        print(f"Epoch {i + 1}")
        train_one_epoch(data, targets, model, optimizer, loss_fn, config.DEVICE, scheduler=scheduler)


if __name__ == "__main__":
    train_overfit(train_folder="../../data/images/trajectories_images",
                  json_file="../../data/labels/labels_5kimgs.json",
                  num_kpts=16)