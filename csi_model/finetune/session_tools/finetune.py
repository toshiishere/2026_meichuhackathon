"""
Fine-tune the pretrained ESP-Fi HAR ResNet18 backbone onto a NEW, growing,
variable-class dataset built by build_dataset.py -- separate from
Model Code/loso_train.py and train_full.py, which are for reproducing the
paper's fixed 7-class, 950-sample-window benchmark and don't fit this task
(different class count, different window length, no train_amp/test_amp
split convention).

What's different here vs a plain train-from-scratch run:
- Classes come from --classes (default: Static Walking Falling), never the
  paper's 7. Any other movement in the dataset (Sitting, Standing) is
  ignored, so those folders can be built now and trained on later just by
  adding them to --classes.
- Only the BACKBONE (everything up to the pooled 512-d feature) loads from
  the pretrained checkpoint. The final classifier is a fresh
  Linear(512, num_classes) -- the pretrained checkpoint's own head was
  Linear(512, 7) for a different task and can't be reused.
- Two-phase schedule: first train only the new head with the backbone frozen
  (including pinning BatchNorm in eval mode, so its running stats aren't
  disturbed by this new, differently-distributed data), then unfreeze and
  fine-tune the whole model at a lower learning rate.
- Train/val split is done on ORIGINAL LABELED INTERVALS (via manifest.csv's
  interval_id), not individual window files -- windows overlap 50% by
  default, so splitting on windows would put near-duplicate samples in both
  train and val and inflate validation metrics.
- Class-weighted loss, since a single session's data is realistically going
  to be imbalanced (e.g. far more no_move than fall).

Usage:
    python finetune.py --data-dir ./built_dataset --checkpoint ../Model\\ Code/pretrained/ResNet18_full.pth \\
        --out ./finetuned --epochs-frozen 10 --epochs-finetune 30
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_mat_amplitude(path):
    mat = sio.loadmat(path)
    x = mat["CSIamp"]
    if x.shape[1] != 52:
        if x.shape[0] == 52:
            x = x.T
        else:
            raise ValueError(f"Unexpected shape {x.shape} in {path}")
    return normalize_amplitude(x)


def normalize_amplitude(x):
    """Shared per-window normalization for training and streaming inference."""
    # scipy's MAT loader returns column-major arrays. Match its reduction order
    # for live windows too, preserving the normalization used by saved models.
    x = np.asfortranarray(x)
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    return x.astype(np.float32)  # (T, 52)


class WindowDataset(Dataset):
    def __init__(self, files, labels, class_to_idx):
        self.files = files
        self.labels = [class_to_idx[l] for l in labels]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = load_mat_amplitude(self.files[idx])
        t = x.shape[0]
        x = torch.from_numpy(x.reshape(1, t, 52))
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def split_by_interval(manifest, val_fraction, seed):
    """Split on (session_id, interval_id) -- never on individual window files,
    so overlapping windows from the same labeled interval stay together on
    one side of the split."""
    rng = np.random.RandomState(seed)
    intervals = manifest[["session_id", "interval_id", "label"]].drop_duplicates()

    train_ids, val_ids = set(), set()
    for label, group in intervals.groupby("label"):
        ids = group["interval_id"].tolist()
        rng.shuffle(ids)
        n_val = max(1, round(len(ids) * val_fraction)) if len(ids) > 1 else 0
        val_ids.update(ids[:n_val])
        train_ids.update(ids[n_val:])

    train_mask = manifest["interval_id"].isin(train_ids)
    val_mask = manifest["interval_id"].isin(val_ids)
    return manifest[train_mask], manifest[val_mask]


def build_model_with_new_head(checkpoint_path, num_classes, model_code_dir):
    sys.path.insert(0, model_code_dir)
    from ESP_Fi_model import ESP_Fi_ResNet18

    # pretrained checkpoint's own head was for a different class count --
    # build the model with a placeholder fc, load everything else, discard
    # the checkpoint's fc, then attach a fresh one for THIS task's classes.
    model = ESP_Fi_ResNet18(num_classes=1)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    filtered = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    expected_missing = {"fc.weight", "fc.bias"}
    if set(missing) - expected_missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch -- missing: {missing}, unexpected: {unexpected}")

    feat_dim = model.fc.in_features
    model.fc = nn.Linear(feat_dim, num_classes)
    return model


def set_backbone_trainable(model, trainable):
    for name, p in model.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = trainable
    # keep BatchNorm running stats frozen too while the backbone is frozen
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)) and not name.startswith("fc"):
            m.eval() if not trainable else m.train()


def evaluate(model, loader, criterion, device, class_names):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            all_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    f1 = f1_score(all_labels, all_preds, labels=list(range(len(class_names))), average="macro", zero_division=0)
    # labels= keeps the report valid when a class (e.g. a rare Falling) has no
    # val samples; without it sklearn raises on the target_names length mismatch
    report = classification_report(all_labels, all_preds, labels=list(range(len(class_names))),
                                    target_names=class_names, zero_division=0, digits=3)
    return acc, f1, total_loss / len(loader.dataset), report


def train_phase(model, train_loader, val_loader, criterion, device, epochs, lr,
                 class_names, best_f1, ckpt_path, phase_name):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        # re-pin any frozen BatchNorm to eval() -- model.train() above would
        # otherwise re-enable running-stat updates on frozen backbone BN layers
        for name, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)) and not name.startswith("fc"):
                if not any(p.requires_grad for p in m.parameters()):
                    m.eval()

        running_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            correct += (torch.argmax(out, dim=1) == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        val_acc, val_f1, val_loss, _ = evaluate(model, val_loader, criterion, device, class_names)
        print(f"[{phase_name}] epoch {epoch+1}/{epochs}: "
              f"train_acc={train_acc:.3f} train_loss={running_loss/total:.4f}  "
              f"val_acc={val_acc:.3f} val_f1={val_f1:.3f} val_loss={val_loss:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)
    return best_f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="output of build_dataset.py")
    p.add_argument("--classes", nargs="+", default=["Static", "Walking", "Falling"],
                    help="movements to train on; other movement folders are ignored")
    p.add_argument("--checkpoint", required=True, help="pretrained single-link ResNet18 .pth")
    p.add_argument("--out", default="./finetuned")
    p.add_argument("--model-code-dir", default="../Model Code")
    p.add_argument("--epochs-frozen", type=int, default=10)
    p.add_argument("--epochs-finetune", type=int, default=30)
    p.add_argument("--lr-frozen", type=float, default=1e-3)
    p.add_argument("--lr-finetune", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest = pd.read_csv(os.path.join(args.data_dir, "manifest.csv"))
    ignored = sorted(set(manifest["label"]) - set(args.classes))
    if ignored:
        print(f"Ignoring movements not in --classes: {ignored}")
    manifest = manifest[manifest["label"].isin(args.classes)]

    present = set(manifest["label"])
    class_names = [c for c in args.classes if c in present]
    missing = [c for c in args.classes if c not in present]
    if missing:
        print(f"WARNING: no data for requested classes {missing}; training on {class_names} only")
    if len(class_names) < 2:
        raise SystemExit(f"Need data for at least 2 classes, found {class_names}")
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print(f"Training classes: {class_names}")

    train_m, val_m = split_by_interval(manifest, args.val_fraction, args.seed)
    print(f"Intervals: {manifest['interval_id'].nunique()} total -> "
          f"{train_m['interval_id'].nunique()} train, {val_m['interval_id'].nunique()} val")
    print(f"Windows: {len(train_m)} train, {len(val_m)} val")

    for split_name, m in [("train", train_m), ("val", val_m)]:
        counts = m["label"].value_counts().to_dict()
        print(f"  {split_name} class counts: {counts}")
        thin = {c: n for c, n in counts.items() if n < 5}
        if thin:
            print(f"  WARNING: very few {split_name} samples for {thin} -- "
                  f"results for these classes will be unreliable until more data is collected")

    def files_for(m, root):
        return [os.path.join(root, row.label, row.filename) for row in m.itertuples()]

    train_files = files_for(train_m, args.data_dir)
    val_files = files_for(val_m, args.data_dir)
    train_ds = WindowDataset(train_files, train_m["label"].tolist(), class_to_idx)
    val_ds = WindowDataset(val_files, val_m["label"].tolist(), class_to_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # class-weighted loss -- a single/few sessions' worth of data is realistically imbalanced
    train_counts = train_m["label"].value_counts()
    weights = torch.tensor(
        [1.0 / max(train_counts.get(c, 1), 1) for c in class_names], dtype=torch.float32
    )
    weights = weights / weights.sum() * len(class_names)
    print(f"Class weights: {dict(zip(class_names, weights.tolist()))}")

    model = build_model_with_new_head(args.checkpoint, len(class_names), args.model_code_dir)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    ckpt_path = os.path.join(args.out, "finetuned_resnet18.pth")
    best_f1 = 0.0

    print("\n=== Phase 1: frozen backbone, training new head only ===")
    set_backbone_trainable(model, trainable=False)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable}")
    best_f1 = train_phase(model, train_loader, val_loader, criterion, device,
                           args.epochs_frozen, args.lr_frozen, class_names,
                           best_f1, ckpt_path, "frozen")

    print("\n=== Phase 2: unfrozen, fine-tuning whole model ===")
    set_backbone_trainable(model, trainable=True)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable}")
    best_f1 = train_phase(model, train_loader, val_loader, criterion, device,
                           args.epochs_finetune, args.lr_finetune, class_names,
                           best_f1, ckpt_path, "finetune")

    print(f"\nBest val macro-F1: {best_f1:.3f}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, _, _, report = evaluate(model, val_loader, criterion, device, class_names)
    print("\nFinal validation report (best checkpoint):")
    print(report)

    with open(os.path.join(args.out, "classes.json"), "w") as f:
        json.dump({"class_names": class_names, "class_to_idx": class_to_idx}, f, indent=2)
    print(f"\nSaved checkpoint: {ckpt_path}")
    print(f"Saved class mapping: {os.path.join(args.out, 'classes.json')}")


if __name__ == "__main__":
    main()
