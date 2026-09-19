"""
Train one model on (almost) the full ESP-Fi HAR dataset and save its weights.

This is NOT for reproducing the paper's Table 4 (that's loso_train.py's job,
using full 8-fold leave-one-subject-out evaluation). This is for producing a
single deployable/transferable checkpoint -- e.g. as the pretrained backbone
for ESP_Fi_ResNet18_MultiLink -- while still holding out ONE participant
(the same participant ID in every environment) as a genuine validation set,
so "best model" selection and progress are based on held-out accuracy rather
than training accuracy alone (train accuracy on a dataset this small reaches
~100% quickly and stops being informative about generalization).

Usage:
    python train_full.py --model ResNet18 --root /path/to/extracted_dataset \
        --out ./pretrained --envs 1 2 3 4 --val-participant 8
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loso_data  # noqa: E402
from loso_train import MODEL_EPOCHS, MODEL_BATCH, build_model, evaluate  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_EPOCHS.keys()))
    p.add_argument("--root", required=True, help="dir with EnvironmentNo.X(...) folders")
    p.add_argument("--out", default="./pretrained")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--envs", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--val-participant", type=int, default=8,
                    help="participant ID (1-8) held out as validation in every "
                         "selected environment; trained on the other 7")
    p.add_argument("--model-code-dir", default=None)
    args = p.parse_args()

    if args.model_code_dir:
        sys.path.insert(0, args.model_code_dir)

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

    epochs = args.epochs or MODEL_EPOCHS[args.model]
    batch_size = MODEL_BATCH[args.model]

    # Pool the 7 non-held-out participants from every requested environment
    # into training, and the held-out participant (same ID in every env) into
    # validation.
    train_items, val_items = [], []
    for env_id in args.envs:
        by_participant = loso_data.load_environment(args.root, env_id)
        if args.val_participant not in by_participant:
            raise ValueError(
                f"--val-participant {args.val_participant} not found in "
                f"environment {env_id} (available: {sorted(by_participant.keys())})"
            )
        for p_id, items in by_participant.items():
            (val_items if p_id == args.val_participant else train_items).extend(items)

    train_ds = loso_data.ListDataset(train_items)
    val_ds = loso_data.ListDataset(val_items)
    print(f"Train set: {len(train_ds)} samples (participants != {args.val_participant}) "
          f"across envs {args.envs}")
    print(f"Val set:   {len(val_ds)} samples (participant {args.val_participant} only)")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model(args.model, num_classes=7).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    ckpt_path = os.path.join(args.out, f"{args.model}_full.pth")
    best_val_acc, best_state = 0.0, None
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        correct = torch.zeros((), device=device)
        total = 0
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum()
            total += labels.size(0)
            running_loss += loss.item() * inputs.size(0)
        train_acc = (correct / total).item()

        val_acc, val_f1, val_loss = evaluate(model, val_loader, criterion, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)  # persist as soon as we have a new best
        scheduler.step()
        print(f"epoch {epoch+1}/{epochs}: train_acc={train_acc:.4f} "
              f"train_loss={running_loss/total:.4f}  "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_loss={val_loss:.4f}  "
              f"best_val={best_val_acc:.4f}")

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s. Best validation accuracy: {best_val_acc:.4f}")
    print(f"(validation = participant {args.val_participant}, held out from training in every environment)")
    print(f"Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
