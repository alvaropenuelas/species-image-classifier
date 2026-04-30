"""
Training loop with early stopping, LR scheduling, and checkpoint saving.
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def _set_train_mode(model):
    model.train()


def _set_inference_mode(model):
    # Switch model to inference mode (disables dropout, batchnorm uses running stats)
    model.eval()  # noqa: S307 — this is nn.Module.eval(), not the builtin


def train(model, train_loader, val_loader, num_epochs, output_dir, device, lr=1e-3, patience=5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    model.to(device)
    best_val_acc = 0.0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # --- train ---
        _set_train_mode(model)
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()

        # --- validate ---
        _set_inference_mode(model)
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        elapsed = time.time() - t0

        print(f"Epoch {epoch:03d}/{num_epochs}  "
              f"train_loss={train_loss/train_total:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss/val_total:.4f}  val_acc={val_acc:.4f}  "
              f"({elapsed:.1f}s)")

        history.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc,
                        "train_loss": train_loss/train_total, "val_loss": val_loss/val_total})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best val_acc={best_val_acc:.4f})")
                break

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val_acc: {best_val_acc:.4f} — model saved to {output_dir}/best_model.pt")
    return best_val_acc
