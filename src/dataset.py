"""
Loads species images from a folder tree: data/raw/<species_name>/<image>.jpg
Applies train/val split and standard augmentation for transfer learning.
"""

import os
from pathlib import Path

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_dataloaders(data_dir: str, val_split: float = 0.2, batch_size: int = 32, num_workers: int = 4):
    data_dir = Path(data_dir)

    full_dataset = datasets.ImageFolder(data_dir, transform=TRAIN_TRANSFORMS)
    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])

    # Apply val transforms to the validation split without touching train_set
    val_set.dataset = datasets.ImageFolder(data_dir, transform=VAL_TRANSFORMS)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    class_names = full_dataset.classes
    return train_loader, val_loader, class_names
