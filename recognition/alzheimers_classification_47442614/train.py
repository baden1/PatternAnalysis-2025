"""
train.py
Contains the training loop for the ConvNeXt model on the ADNI dataset.
"""

import torch
from torchvision.models import convnext_small
from dataset import ADNI_Dataset, train_transform, test_transform
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import os

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 50

# datasets / dataloaders
train_dataset = ADNI_Dataset(root_dir=os.path.join('dataset', 'AD_NC'), split="train", transform=train_transform)
test_dataset = ADNI_Dataset(root_dir=os.path.join('dataset', 'AD_NC'), split="test", transform=test_transform)

# split test dataset into validation and test sets
test_size = int(0.5 * len(test_dataset))
val_size = len(test_dataset) - test_size
test_dataset, val_dataset = random_split(test_dataset, [test_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# model
model = convnext_small(pretrained=False, num_classes=2)
model = model.to(device)

# loss function
criterion = nn.CrossEntropyLoss()

# optimiser
optimiser = optim.AdamW(model.parameters(), lr=learning_rate)

train_losses = []
val_losses = []

# training loop
for epoch in range(num_epochs):
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

    print(f"Epoch {epoch+1}/{num_epochs} "
          f"Train Loss: {epoch_train_loss:.4f} "
          f"Val Loss: {epoch_val_loss:.4f} "
          f"Val Acc: {val_accuracy:.4f}")
    
# save model
torch.save(model.state_dict(), "convnext.pth")
    
# plot training and validation loss
plt.figure(figsize=(8,5))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
os.makedirs('figs', exist_ok=True)
plt.savefig(os.path.join('figs', 'loss_curve.png'))
