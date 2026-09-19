"""
Minimal inference script for the ESP-Fi HAR ResNet18 checkpoint.

Loads a trained checkpoint and runs it on one or more .mat CSI files,
printing the predicted activity (and, if the filename follows the dataset's
X-Y-Z-M naming convention, whether it matches the true label).

Usage:
    python infer.py                              # runs on everything in sample_data/
    python infer.py --input sample_data/2-8-2-1.mat
    python infer.py --input /path/to/some/dir    # runs on every .mat file in a dir
"""
import argparse
import glob
import os
import re

import numpy as np
import scipy.io as sio
import torch

from ESP_Fi_model import ESP_Fi_ResNet18

ACTIVITIES = ["arm_wave", "fall", "jump", "run", "squat", "turn", "walk"]
ACTIVITY_ID_TO_NAME = {
    1: "run", 2: "fall", 3: "walk", 4: "turn",
    5: "jump", 6: "squat", 7: "arm_wave",
}
FNAME_RE = re.compile(r"(\d+)-(\d+)-(\d+)-(\d+)\.mat$")


def load_and_preprocess(mat_path):
    """Same preprocessing the training pipeline uses: load CSIamp, orient to
    (950, 52), z-score normalize per sample, reshape to (1, 1, 950, 52)."""
    mat = sio.loadmat(mat_path)
    x = mat["CSIamp"]
    if x.shape == (950, 52):
        pass
    elif x.shape == (52, 950):
        x = x.T
    else:
        raise ValueError(f"Unexpected CSIamp shape {x.shape} in {mat_path}")
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    x = x.reshape(1, 1, 950, 52).astype(np.float32)
    return torch.from_numpy(x)


def true_label_from_filename(path):
    m = FNAME_RE.search(os.path.basename(path))
    if not m:
        return None
    _, _, activity_id, _ = map(int, m.groups())
    return ACTIVITY_ID_TO_NAME.get(activity_id)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="pretrained/ResNet18_full.pth")
    p.add_argument("--input", default="sample_data")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ESP_Fi_ResNet18(num_classes=7)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}\n")

    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*.mat")))
    else:
        files = [args.input]

    if not files:
        print(f"No .mat files found at {args.input}")
        return

    correct, total_with_label = 0, 0
    for f in files:
        x = load_and_preprocess(f).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs))
        pred_name = ACTIVITIES[pred_idx]
        confidence = float(probs[pred_idx])

        true_name = true_label_from_filename(f)
        marker = ""
        if true_name is not None:
            total_with_label += 1
            is_correct = pred_name == true_name
            correct += int(is_correct)
            marker = f"  (true: {true_name}, {'correct' if is_correct else 'WRONG'})"

        print(f"{os.path.basename(f):20s} -> predicted: {pred_name:10s} "
              f"(confidence {confidence:.1%}){marker}")

    if total_with_label:
        print(f"\n{correct}/{total_with_label} correct "
              f"({correct/total_with_label:.1%}) on files with recoverable ground truth")


if __name__ == "__main__":
    main()
