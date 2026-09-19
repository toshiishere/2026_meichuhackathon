"""
Standalone GPU-utilization / throughput sweep across batch sizes for
ESP_Fi_ResNet18 (or any model in ESP_Fi_model.py). Does NOT modify
loso_train.py or its MODEL_BATCH settings -- purely diagnostic.

Run this on the actual training machine (with the real GPU):
    python3 batch_size_sweep.py --root /path/to/extracted_dataset --model ResNet18

It loads one real LOSO fold's data (so tensor shapes/sizes match production),
then times N training steps at each candidate batch size and reports
samples/sec -- the batch size where samples/sec stops improving much is your
practical ceiling; going further only burns memory for no speed benefit.
"""
import argparse
import statistics
import sys
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loso_data  # noqa: E402


def build_model(name, num_classes):
    from ESP_Fi_model import CNN, ESP_Fi_ResNet18
    return {"CNN": CNN, "ResNet18": ESP_Fi_ResNet18}[name](num_classes)


def bench_batch_size(model_name, dataset, batch_size, device, steps=15, warmup=5):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = build_model(model_name, num_classes=7).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    model.train()

    it = iter(loader)
    times = []
    for i in range(warmup + steps):
        try:
            inputs, labels = next(it)
        except StopIteration:
            it = iter(loader)
            inputs, labels = next(it)
        inputs, labels = inputs.to(device), labels.to(device).long()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        opt.zero_grad()
        out = model(inputs)
        loss = crit(out, labels)
        loss.backward()
        opt.step()

        if device.type == "cuda":
            torch.cuda.synchronize()  # honest timing: wait for the GPU to actually finish
        dt = time.time() - t0

        if i >= warmup:
            times.append(dt)

    median_dt = statistics.median(times)
    samples_per_sec = batch_size / median_dt
    return median_dt, samples_per_sec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--model", default="ResNet18", choices=["CNN", "ResNet18"])
    p.add_argument("--env", type=int, default=1)
    p.add_argument("--batch-sizes", type=int, nargs="+",
                    default=[8, 16, 32, 64, 128, 256, 490])
    p.add_argument("--model-code-dir", default=None,
                    help="dir containing ESP_Fi_model.py, if not already on PYTHONPATH")
    args = p.parse_args()

    if args.model_code_dir:
        sys.path.insert(0, args.model_code_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

    by_participant = loso_data.load_environment(args.root, args.env)
    train_ds, _ = loso_data.loso_fold(by_participant, held_out=sorted(by_participant.keys())[0])
    print(f"Fold train set size: {len(train_ds)} samples\n")

    print(f"{'batch_size':>10} | {'ms/step':>10} | {'samples/sec':>12} | {'speedup vs bs=8':>16}")
    print("-" * 56)
    base_rate = None
    for bs in args.batch_sizes:
        if bs > len(train_ds):
            print(f"{bs:>10} | skipped (larger than fold's {len(train_ds)} samples)")
            continue
        dt, rate = bench_batch_size(args.model, train_ds, bs, device)
        if base_rate is None:
            base_rate = rate
        print(f"{bs:>10} | {dt*1000:>10.2f} | {rate:>12.1f} | {rate/base_rate:>15.2f}x")


if __name__ == "__main__":
    main()
