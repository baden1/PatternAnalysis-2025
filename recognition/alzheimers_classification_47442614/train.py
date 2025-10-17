"""
train.py
Contains the training loop for the ConvNeXt model on the ADNI dataset.
"""

import torch
from dataset import ADNI_Dataset, train_transform, test_transform
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import os
import pandas as pd
from modules import convnext_small

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 50

# early stopping parameters
patience = 10
best_val_loss = float("inf")
epochs_without_improvement = 0
early_stop = False

path = "/home/groups/comp3710/ADNI/AD_NC"

# datasets / dataloaders
train_dataset = ADNI_Dataset(
    root_dir=path, split="train", transform=train_transform
)
test_dataset = ADNI_Dataset(
    root_dir=path, split="test", transform=test_transform
)

# split train dataset into validation and train sets
val_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - val_size

# rng generator - set seed for reproducibility
generator = torch.Generator().manual_seed(0)
train_dataset, val_dataset = random_split(test_dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# model
model = convnext_small(num_classes=2)

# loss function
criterion = nn.CrossEntropyLoss()

# optimiser
optimiser = optim.AdamW(model.parameters(), lr=learning_rate)

# learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser,
    mode="min",  # minimise validation loss
    factor=0.1,  # reduce lr by a factor of 10
    patience=patience,  # wait 10 epochs with no improvement before reducing
    min_lr=1e-7,  # stop reducing below this LR
)

train_losses = []
val_losses = []
lr_history = []

# training loop
for epoch in range(num_epochs):
    print(f'starting epoch {epoch+1}/{num_epochs}')
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
        running_loss += loss.item() * images.size(0)

    # average training loss for the epoch
    epoch_train_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_train_loss)

    # validation
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_val_loss = running_val_loss / len(val_dataset)
    val_losses.append(epoch_val_loss)
    val_accuracy = correct / total

    # determine whether to adjust learning rate
    scheduler.step(epoch_val_loss)
    current_lr = optimiser.param_groups[0]["lr"]
    lr_history.append(current_lr)

    print(
        f"Epoch {epoch+1}/{num_epochs} "
        f"Train Loss: {epoch_train_loss:.4f} "
        f"Val Loss: {epoch_val_loss:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )

    # save model with the best validation loss
    if epoch_val_loss < best_val_loss:
        torch.save(model.state_dict(), "convnext.pth")

# save loss and lr history to csv
os.makedirs("logs", exist_ok=True)
df = pd.DataFrame({
    'train_loss': train_losses,
    'val_loss': val_losses,
    'learning_rate': lr_history
})
df.to_csv(os.path.join('logs', 'training_log2.csv'), index_label='epoch')

# plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(range(len(train_losses)), train_losses, label="Train Loss")
plt.plot(range(len(val_losses)), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
os.makedirs("figs", exist_ok=True)
plt.savefig(os.path.join("figs", "loss_curve.png"))


# plot learning rate history
plt.clf()
plt.semilogy(range(len(lr_history)), lr_history)
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("ReduceLROnPlateau Learning Rate History")
plt.savefig(os.path.join("figs", "lr_history.png"))
