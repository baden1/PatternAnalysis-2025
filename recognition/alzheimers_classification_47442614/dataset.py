"""
dataset.py
Contains logic for creation of the data loader for loading and pre-processing
the ADNI dataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms


class ADNI_Dataset(Dataset):
    """Custom dataset class for loading images and labels from the ADNI dataset."""

    def __init__(self, root_dir, split="train", transform=None):
        """
        Initialise the ADNI dataset.

        Args:
            root_dir (str): Root directory of dataset.
            split (str): One of ["train", "test"].
            transform: torchvision transforms to apply to images.
        """

        # either train or test split
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform

        # class names and integer labels
        self.classes = ["AD", "NC"]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.image_paths = []
        self.labels = []

        # save image paths and their labels
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                    # add image paths and labels if they are valid images
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        """Get size of dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get image and label at the given index.

        Args:
            idx (int): index

        Returns:
            (PIL.Image, int): Image and label id. 
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# -- Transforms --

# train transform - data augmentation and normalisation for training
train_transform = transforms.Compose([
        # Convert grayscale to 3-channel by duplicating channels
        transforms.Grayscale(num_output_channels=3),

        # resize to standard size for ConvNeXt
        transforms.Resize((256, 256)),

        # small random crop to 224x224 size
        transforms.RandomResizedCrop(224, scale=(0.95, 1.0)),

        # small random affine transformations
        transforms.RandomAffine(
            degrees=5,  # rotation
            translate=(0.02, 0.02),  # translation
            scale=(0.95, 1.05),  # scaling
        ),

        # slight intensity variations
        transforms.ColorJitter(
            brightness=0.1, contrast=0.1
        ),

        # convert to tensor and normalise
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        ),
])

# test transform - more simple transforms for testing
test_transform = transforms.Compose([
    # Convert grayscale to 3-channel by duplicating channels
    transforms.Grayscale(num_output_channels=3),  
    
    # resize to 224 x 224 for ConvNeXt
    transforms.Resize((224, 224)), 

    # convert to tensor and normalise
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
