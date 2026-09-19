"""Session fine-tuning entry point; reuses the supplied ESP-Fi model and loader."""

import json
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from finetune import (
    WindowDataset,
    build_model_with_new_head,
    set_backbone_trainable,
    evaluate,
)


def interval_split(manifest, fraction=0.2, seed=42):
    """All receivers and overlapping windows from one labeled event stay together."""
    rng = np.random.default_rng(seed)
    validation = set()
    events = manifest[["session_id", "interval_id", "label"]].drop_duplicates()
    for _, group in events.groupby("label"):
        keys = list(zip(group.session_id, group.interval_id))
        rng.shuffle(keys)
        if len(keys) >= 2:
            validation.update(
                keys[: min(len(keys) - 1, max(1, round(len(keys) * fraction)))]
            )
    mask = np.array(
        [
            (s, i) in validation
            for s, i in zip(manifest.session_id, manifest.interval_id)
        ]
    )
    return manifest[~mask].copy(), manifest[mask].copy()


def train_session(
    dataset,
    checkpoint,
    output,
    model_code,
    epochs_frozen=5,
    epochs_finetune=15,
    batch_size=8,
    seed=42,
):
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError(
            "Fine-tuning requires an available ROCm GPU; CPU fallback is disabled"
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    manifest = pd.read_csv(dataset / "manifest.csv")
    classes = sorted(manifest.label.unique().tolist())
    if len(classes) < 2:
        raise ValueError(
            f"Need at least two labeled actions with usable CSI windows; found {classes}"
        )
    train, val = interval_split(manifest, seed=seed)
    mapping = {label: index for index, label in enumerate(classes)}

    def loader(rows, shuffle):
        paths = [str(dataset / row.label / row.filename) for row in rows.itertuples()]
        return DataLoader(
            WindowDataset(paths, rows.label.tolist(), mapping),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    train_loader, val_loader = loader(train, True), loader(val, False)
    model = build_model_with_new_head(
        str(checkpoint), len(classes), str(model_code)
    ).to(device)
    counts = train.label.value_counts()
    weights = torch.tensor(
        [1 / counts[label] for label in classes], dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=weights / weights.mean())
    history, best = [], float("-inf")
    checkpoint_out = output / "finetuned_resnet18.pth"
    if val.empty:
        print(
            "No independent validation intervals: training only; no validation score will be reported.",
            flush=True,
        )
    for phase, epochs, lr in [
        ("head", epochs_frozen, 1e-3),
        ("finetune", epochs_finetune, 1e-4),
    ]:
        set_backbone_trainable(model, phase == "finetune")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
        )
        for epoch in range(epochs):
            model.train()
            if phase == "head":
                set_backbone_trainable(
                    model, False
                )  # Keep pretrained BN statistics frozen.
            loss_sum, total = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(y)
                total += len(y)
            record = dict(phase=phase, epoch=epoch + 1, train_loss=loss_sum / total)
            if not val.empty:
                acc, f1, val_loss, _ = evaluate(
                    model, val_loader, criterion, device, classes
                )
                record.update(val_accuracy=acc, val_macro_f1=f1, val_loss=val_loss)
                score = f1
            else:
                # With no holdout, publish the final epoch, not a fabricated validation metric.
                score = len(history)
            if score > best or val.empty:
                best = score
                torch.save(
                    {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    checkpoint_out,
                )
            history.append(record)
            print(json.dumps(record), flush=True)
    if not checkpoint_out.exists():
        raise ValueError("At least one training epoch is required")
    train.assign(split="train").to_csv(output / "train_split.csv", index=False)
    val.assign(split="validation").to_csv(output / "validation_split.csv", index=False)
    (output / "classes.json").write_text(
        json.dumps(dict(class_names=classes, class_to_idx=mapping), indent=2)
    )
    metrics = dict(
        history=history,
        train_windows=len(train),
        validation_windows=len(val),
        validation_available=not val.empty,
        best_validation_macro_f1=best if not val.empty else None,
        validation_missing_classes=sorted(set(classes) - set(val.label)),
        seed=seed,
        gpu=torch.cuda.get_device_name(),
        rocm=torch.version.hip,
        warning="Labels are automatic predictions; interval holdout is not evaluation on a new session/person.",
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    return metrics
