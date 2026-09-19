"""
Full LOSO (leave-one-subject-out) reproduction of ESP-Fi HAR paper's Table 4.

Usage:
    python loso_train.py --model ResNet18 --root /path/to/extracted_dataset \
        --out ./loso_results --epochs 50

--root should point at the directory containing the four
"EnvironmentNo.X(...)"" folders (i.e. the extracted ESP-Fi_dataset.rar).

Requires ESP_Fi_model.py to be importable (same dir as ESP-Fi HAR's Model Code,
or pass --model-code-dir to point at it) and loso_data.py (shipped alongside
this script).
"""
import argparse
import csv
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

MODEL_EPOCHS = {
    "CNN": 50, "ResNet18": 50, "MobileNetV3": 50, "EfficientNetLite": 50,
    "Transformer": 100, "GRU": 100, "LSTM": 100,
}
MODEL_BATCH = {
    "GRU": 64,
    "CNN": 32, "LSTM": 32, "ResNet18": 32, "MobileNetV3": 32, "EfficientNetLite": 32,
    "Transformer": 4,
}


def build_model(name, num_classes):
    from ESP_Fi_model import (
        CNN, ESP_Fi_ResNet18, ESP_Fi_Transformer, ESP_Fi_GRU, ESP_Fi_LSTM,
        MobileNetV3, EfficientNetLite,
    )
    return {
        "CNN": CNN, "ResNet18": ESP_Fi_ResNet18, "Transformer": ESP_Fi_Transformer,
        "GRU": ESP_Fi_GRU, "LSTM": ESP_Fi_LSTM, "MobileNetV3": MobileNetV3,
        "EfficientNetLite": EfficientNetLite,
    }[name](num_classes)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1, total_loss / len(loader.dataset)


def train_one_fold(model_name, train_ds, test_ds, epochs, batch_size, device, checkpoint_path=None):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = build_model(model_name, num_classes=7).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_train_acc, best_state = 0.0, None
    for _ in range(epochs):
        model.train()
        correct = torch.zeros((), device=device)
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum()  # stays on-device, no sync
            total += labels.size(0)
        train_acc = (correct / total).item()  # single sync per epoch, not per batch
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            # clone() on-device: no host transfer, no forced sync
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        scheduler.step()

    model.load_state_dict(best_state)
    if checkpoint_path:
        torch.save(best_state, checkpoint_path)
    return evaluate(model, test_loader, criterion, device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                    choices=list(MODEL_EPOCHS.keys()))
    p.add_argument("--root", required=True, help="dir with EnvironmentNo.X(...) folders")
    p.add_argument("--out", default="./loso_results")
    p.add_argument("--epochs", type=int, default=None, help="override default epoch count")
    p.add_argument("--envs", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--model-code-dir", default=None,
                    help="dir containing ESP_Fi_model.py, if not already on PYTHONPATH")
    args = p.parse_args()

    if args.model_code_dir:
        sys.path.insert(0, args.model_code_dir)

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

    epochs = args.epochs or MODEL_EPOCHS[args.model]
    batch_size = MODEL_BATCH[args.model]

    csv_path = os.path.join(args.out, f"loso_{args.model}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env", "held_out_participant", "acc", "f1", "loss", "seconds"])

    env_names = {1: "Corridor", 2: "Office", 3: "Meeting Room", 4: "Laboratory"}
    summary = {}

    for env_id in args.envs:
        print(f"\n=== Environment {env_id} ({env_names[env_id]}) ===")
        by_participant = loso_data.load_environment(args.root, env_id)
        fold_accs, fold_f1s = [], []

        for held_out in sorted(by_participant.keys()):
            train_ds, test_ds = loso_data.loso_fold(by_participant, held_out)
            ckpt_path = os.path.join(
                args.out, f"{args.model}_env{env_id}_heldout{held_out}.pth"
            )
            t0 = time.time()
            acc, f1, loss = train_one_fold(
                args.model, train_ds, test_ds, epochs, batch_size, device,
                checkpoint_path=ckpt_path,
            )
            dt = time.time() - t0
            fold_accs.append(acc)
            fold_f1s.append(f1)
            print(f"  fold(held_out=participant {held_out}): acc={acc:.4f} f1={f1:.4f} ({dt:.1f}s)")
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([env_id, held_out, acc, f1, loss, dt])

        mean_acc, std_acc = np.mean(fold_accs) * 100, np.std(fold_accs) * 100
        mean_f1 = np.mean(fold_f1s) * 100
        summary[env_names[env_id]] = (mean_acc, std_acc, mean_f1)
        print(f"  -> {env_names[env_id]}: Acc {mean_acc:.2f}±{std_acc:.2f}%  F1 {mean_f1:.2f}%")

    print(f"\n=== Summary (compare to paper's Table 4, model={args.model}) ===")
    for env, (acc, std, f1) in summary.items():
        print(f"{env:15s} Acc {acc:6.2f}% (std {std:5.2f})  F1 {f1:6.2f}%")


if __name__ == "__main__":
    main()
