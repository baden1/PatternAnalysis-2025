from dataset import ADNI_Dataset, train_transform, test_transform
import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 32

path = "/home/groups/comp3710/ADNI/AD_NC"

# datasets / dataloaders
train_dataset = ADNI_Dataset(
    root_dir=path, split="train", transform=train_transform
)
test_dataset = ADNI_Dataset(
    root_dir=path, split="test", transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# load model
model = torch.load("convnext.pth")
model = model.to(device)
model.eval()

results = []

correct = 0
total = 0
for images, labels in test_loader:
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for image, label in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                results.append({
                    "true_label": label,
                    "predicted_label": predicted.cpu().numpy(),
                })

# save results
df = pd.DataFrame(results)
df.to_csv("test_results.csv", index=True, index_label="image")

# evaluate accuracy
print(f"Test Accuracy: {100 * correct / total:.2f}%")