"""
Plots training/validation accuracy and loss curves from outputs/history.json.
Saves figure to outputs/training_curves.png.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_history(history_path: str = "outputs/history.json", out_path: str = "outputs/training_curves.png"):
    with open(history_path) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("EfficientNet-B0 — Coastal Species Classifier", fontsize=13, fontweight="bold")

    # Accuracy
    ax1.plot(epochs, train_acc, "o-", color="#2196F3", label="Train", linewidth=2, markersize=4)
    ax1.plot(epochs, val_acc, "s--", color="#4CAF50", label="Validation", linewidth=2, markersize=4)
    ax1.axhline(max(val_acc), color="#4CAF50", linewidth=0.7, linestyle=":", alpha=0.7)
    ax1.text(epochs[-1], max(val_acc) + 0.005, f"best={max(val_acc):.2%}", ha="right", fontsize=9, color="#4CAF50")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Loss
    ax2.plot(epochs, train_loss, "o-", color="#2196F3", label="Train", linewidth=2, markersize=4)
    ax2.plot(epochs, val_loss, "s--", color="#FF5722", label="Validation", linewidth=2, markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-entropy loss")
    ax2.set_title("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    Path(out_path).parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_history()
