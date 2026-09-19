"""One subprocess per session job; outputs are isolated until each stage succeeds."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from apps.common.storage import atomic_json, utc_now
from .preprocess import build_dataset

ROOT = Path(__file__).resolve().parents[3]
PRETRAINED = ROOT / "csi_model/Model Code/pretrained/ResNet18_full.pth"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def publish(source, destination):
    temporary = destination.with_name("." + destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def run(session, run_dir, options):
    def stage(name):
        atomic_json(run_dir / "progress.json", dict(stage=name, updated_at=utc_now()))
        print(f"=== {name} ===", flush=True)

    import torch

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU unavailable; CPU fallback is disabled")
    action = options["action"]
    train = session / "train"
    if action in {"label", "auto"}:
        stage("labeling")
        from pose_labeling.inference import run_text_inference

        run_text_inference(session / "raw/video.mp4", run_dir / "action_results.csv")
        publish(run_dir / "action_results.csv", train / "action_results.csv")
        publish(run_dir / "action_results.json", train / "action_results.json")
        # Keep previous model artifacts but mark their labeling lineage explicitly.
    if action in {"finetune", "auto"}:
        stage("preprocessing")
        labels = train / "action_results.csv"
        if not labels.is_file():
            raise ValueError("Run Labeling first: train/action_results.csv is missing")
        snapshot = run_dir / "labels_used.csv"
        shutil.copyfile(labels, snapshot)
        summary = build_dataset(
            session,
            snapshot,
            run_dir / "dataset",
            **{k: options[k] for k in ("window_seconds", "sample_rate_hz", "overlap")},
        )
        stage("fine-tuning")
        copied = run_dir / "pretrained_resnet18.pth"
        shutil.copyfile(PRETRAINED, copied)
        sys.path.insert(0, str(ROOT / "csi_model/finetune/session_tools"))
        from train_session import train_session

        metrics = train_session(
            run_dir / "dataset",
            copied,
            run_dir,
            ROOT / "csi_model/Model Code",
            **{
                k: options[k]
                for k in ("epochs_frozen", "epochs_finetune", "batch_size", "seed")
            },
        )
        metadata = dict(
            run_id=run_dir.name,
            session_id=session.name,
            created_at=utc_now(),
            options=options,
            pretrained_sha256=digest(copied),
            labels_sha256=digest(snapshot),
            model_sha256=digest(run_dir / "finetuned_resnet18.pth"),
            preprocessing=summary,
            architecture="ESP_Fi_ResNet18",
            metrics=metrics,
            model_path=f"train/runs/{run_dir.name}/finetuned_resnet18.pth",
        )
        atomic_json(run_dir / "model.json", metadata)
        for name in (
            "finetuned_resnet18.pth",
            "classes.json",
            "metrics.json",
            "pretrained_resnet18.pth",
        ):
            publish(run_dir / name, train / name)
        # Publish this pointer last. Run-specific model/classes remain a consistent pair.
        atomic_json(train / "model.json", metadata)
    stage("done")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    run(args.session, args.run, json.loads((args.run / "options.json").read_text()))
