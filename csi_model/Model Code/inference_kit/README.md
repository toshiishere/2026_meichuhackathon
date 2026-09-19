# ESP-Fi HAR — Minimal Inference Kit

Run a pretrained WiFi CSI human-activity-recognition model on sample data in a
few minutes. No dataset download or training required — everything you need
is in this folder.

## What this is

A ResNet18 model, trained on the [ESP-Fi HAR dataset](https://github.com/AutoSmartGroup/ESP-Fi-HAR)
(WiFi CSI captured from cheap ESP32 modules) to classify 7 human activities
from a 10-second WiFi CSI capture:

`run`, `fall`, `walk`, `turn`, `jump`, `squat`, `arm_wave`

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

(If you have a GPU with CUDA or ROCm set up, `infer.py` will use it
automatically — no extra steps needed. CPU works fine too, inference on one
sample takes well under a second either way.)

## Run it

```bash
python3 infer.py
```

This runs the model on all 7 files in `sample_data/` (one example per
activity, from a participant the model never saw during training) and prints
predictions with confidence scores:

```
Device: cpu
Loaded checkpoint: pretrained/ResNet18_full.pth

2-8-1-1.mat          -> predicted: run        (confidence 98.7%)  (true: run, correct)
2-8-2-1.mat          -> predicted: fall       (confidence 51.0%)  (true: fall, correct)
2-8-3-1.mat          -> predicted: walk       (confidence 83.8%)  (true: walk, correct)
2-8-4-1.mat          -> predicted: turn       (confidence 79.4%)  (true: turn, correct)
2-8-5-1.mat          -> predicted: jump       (confidence 97.1%)  (true: jump, correct)
2-8-6-1.mat          -> predicted: squat      (confidence 100.0%)  (true: squat, correct)
2-8-7-1.mat          -> predicted: arm_wave   (confidence 90.4%)  (true: arm_wave, correct)

7/7 correct (100.0%) on files with recoverable ground truth
```

To run on a single file or your own `.mat` files instead:

```bash
python3 infer.py --input sample_data/2-8-2-1.mat
python3 infer.py --input /path/to/your/mat/files/   # runs every .mat file in a directory
```

### Input format

Each `.mat` file must contain a `CSIamp` array of shape `(950, 52)` or
`(52, 950)` — WiFi CSI amplitude, 950 time samples × 52 subcarriers, from a
single antenna/link, covering a 10-second capture at ~100 Hz. This is exactly
what the ESP-Fi HAR dataset (and its capture tool, ESP-CSI-Tool on an
ESP32-C3) produces per activity trial. The script applies the same
preprocessing used at training time: per-sample z-score normalization
(`(x - mean) / std`), nothing else.

## About the checkpoint

`pretrained/ResNet18_full.pth` was trained on the full ESP-Fi HAR dataset
(4 indoor environments × 7 participants, pooled), with **participant 8 held
out as a real validation set** (used only to pick the best checkpoint, never
trained on). On that held-out participant: **91.1% accuracy, 0.910 macro-F1**
across all 4 environments.

**Important caveat**: this number is *not* directly comparable to the
accuracy figures reported in the ESP-Fi HAR paper's Table 4 (which are
lower, ~58–70%). The paper's numbers come from 8-fold leave-one-subject-out
cross-validation done *per environment* (averaged over 8 different held-out
participants, trained on only ~490 samples each). This checkpoint instead
pools all 4 environments together for training (1,960 samples) and validates
against just one fixed held-out participant — an easier, less robust
generalization test. Treat 91.1% as "this specific checkpoint works well on
this specific held-out person," not as a paper-comparable benchmark.

## Files in this kit

| File | Purpose |
|---|---|
| `infer.py` | Run this — loads the checkpoint and predicts on `.mat` files |
| `ESP_Fi_model.py` | Model architecture definitions (only `ESP_Fi_ResNet18` is used here) |
| `pretrained/ResNet18_full.pth` | Trained weights |
| `sample_data/*.mat` | 7 demo CSI captures, one per activity, from held-out participant 8 |
| `requirements.txt` | Python dependencies |

## Troubleshooting

- **`ModuleNotFoundError: No module named 'einops'`** — run
  `pip install -r requirements.txt`; `ESP_Fi_model.py` imports `einops` at
  the top even though the ResNet18 path used here doesn't need it directly.
- **Shape errors on your own data** — double check your `.mat` file has a
  `CSIamp` key shaped `(950, 52)`; other shapes/keys aren't supported by
  this script.
