"""
Computes per-class accuracy and a confusion matrix on the validation set.
Saves outputs/confusion_matrix.png and prints a classification report.

Usage:
    python evaluate.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, transforms

from src.model import build_model
from src.train import _set_inference_mode

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def evaluate(data_dir="data/raw", checkpoint="outputs/best_model.pt", classes_path="outputs/classes.json"):
    with open(classes_path) as f:
        class_names = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    _set_inference_mode(model)

    dataset = datasets.ImageFolder(data_dir, transform=VAL_TRANSFORMS)
    # Use a fixed split matching training (last 20%)
    n_val = int(len(dataset) * 0.2)
    _, val_set = torch.utils.data.random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for img, label in torch.utils.data.DataLoader(val_set, batch_size=32):
            img = img.to(device)
            preds = model(img).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(label.tolist())

    # Short names for display
    short_names = [c.replace("_", " ") for c in class_names]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=short_names, digits=3))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalised Confusion Matrix\nCoastal Species Classifier — EfficientNet-B0")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if cm_norm[i, j] > 0.6 else "black")

    plt.tight_layout()
    out = Path("outputs/confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix saved: {out}")


if __name__ == "__main__":
    evaluate()
