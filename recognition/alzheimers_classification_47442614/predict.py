"""
predict.py
Contains logic for running inference on the ConvNeXt model on either
a single image or the whole test dataset.
"""

from dataset import ADNI_Dataset, train_transform, test_transform
import torch
from torch.utils.data import DataLoader, random_split
from modules import convnext_small
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 32

dataset_path = "/home/groups/comp3710/ADNI/AD_NC"

# datasets / dataloaders
test_dataset = ADNI_Dataset(
    root_dir=dataset_path, split="test", transform=test_transform
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# load model
state_dict = torch.load("convnext.pth")
model = convnext_small(num_classes=2)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

def predict_test_set():
    """Predict image labels for every image in the test set."""

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

                for label, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                    results.append({
                        "true_label": label,
                        "predicted_label": pred,
                    })

    # save results
    df = pd.DataFrame(results)
    df.to_csv("test_results.csv", index=True, index_label="image")

    # evaluate accuracy
    print(f"Test Accuracy: {100 * correct / total:.2f}%")


def predict_single_image(image_path):
    """Predicts the label of a single image and displays the image and prediction

    Args:
        image_path (str): Path to the image to predict
    """

    # load image
    image = Image.open(image_path).convert("RGB")

    # apply test transform and turn into size-1 batch
    image_tensor = test_transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # predict
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    pred_index = int(predicted.item())
    pred_label = test_dataset.classes[pred_index]

    plt.imshow(image)
    plt.title(f'Predicted: {pred_label}')
    plt.savefig(os.path.join('figs', 'image_with_prediction.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on the ConvNeXt model.")
    parser.add_argument('--single', type=str, help="Path to a single image for prediction.")
    args = parser.parse_args()

    if args.single:
        predict_single_image(args.single)
    else:
        predict_test_set()
    