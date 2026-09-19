"""Reusable Ryzen AI XINT8 model preparation and inference.

The public functions in this module are deliberately independent of FastAPI so
other Python applications can import the same conversion and runtime path.
"""

import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parents[3]
CACHE_VERSION = 1
CALIBRATION_SAMPLES = 128


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_model(checkpoint: Path, class_count: int):
    import torch

    sys.path.insert(0, str(ROOT / "csi_model/Model Code"))
    from ESP_Fi_model import ESP_Fi_ResNet18

    model = ESP_Fi_ResNet18(num_classes=class_count)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    return model.eval()


def _normalized_window(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    value = scipy.io.loadmat(path)["CSIamp"]
    if value.shape == expected_shape[::-1]:
        value = value.T
    if value.shape != expected_shape:
        raise ValueError(
            f"Calibration window {path.name} has shape {value.shape}, expected {expected_shape}"
        )
    value = np.asarray(value, dtype=np.float32)
    return ((value - value.mean()) / (value.std() + 1e-8)).astype(np.float32)


def calibration_windows(
    checkpoint: Path, classes: list[str], expected_shape: tuple[int, int]
) -> list[np.ndarray]:
    """Select deterministic, class-balanced windows from the model training split."""
    run = checkpoint.parent
    session = checkpoint.parents[3]
    split = run / "train_split.csv"
    dataset = run / "dataset"
    if not dataset.is_dir():
        # Compatibility with sessions trained before datasets were retained per run.
        dataset = session / "train/dataset"
    if not split.is_file() or not dataset.is_dir():
        raise ValueError(
            "NPU conversion needs the model train_split.csv and train/dataset calibration windows"
        )
    by_class = {name: [] for name in classes}
    with split.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("label") in by_class:
                candidate = dataset / row["label"] / row["filename"]
                if candidate.is_file() and not candidate.is_symlink():
                    by_class[row["label"]].append(candidate)
    missing = [name for name, paths in by_class.items() if not paths]
    if missing:
        raise ValueError(
            f"No NPU calibration windows for classes: {', '.join(missing)}"
        )
    # Round-robin classes so an imbalanced training session does not make the
    # activation ranges almost entirely describe the majority class.
    selected = []
    index = 0
    while len(selected) < CALIBRATION_SAMPLES:
        added = False
        for name in classes:
            paths = by_class[name]
            if index < len(paths):
                selected.append(paths[index])
                added = True
                if len(selected) == CALIBRATION_SAMPLES:
                    break
        if not added:
            break
        index += 1
    return [_normalized_window(path, expected_shape) for path in selected]


class CalibrationDataReader:
    """Quark-compatible reader over already normalized CSI windows."""

    def __init__(self, windows: list[np.ndarray]):
        self.windows = windows
        self.rewind()

    def get_next(self):
        try:
            value = next(self.iterator)
        except StopIteration:
            return None
        return {"input": value[None, None].astype(np.float32, copy=False)}

    def rewind(self):
        self.iterator = iter(self.windows)

    reset = rewind


def artifact_paths(checkpoint: Path) -> dict[str, Path]:
    stem = checkpoint.with_suffix("")
    return {
        "onnx": stem.with_suffix(".onnx"),
        "int8": stem.with_name(stem.name + ".int8.onnx"),
        "manifest": stem.with_name(stem.name + ".int8.json"),
        "vaip_cache": stem.with_name(stem.name + ".vaip"),
        "lock": stem.with_name(stem.name + ".npu.lock"),
    }


def prepare_int8_model(
    metadata: dict, checkpoint: Path, classes: list[str]
) -> dict[str, Path]:
    """Export and XINT8-quantize once, atomically, beside the PyTorch checkpoint."""
    paths = artifact_paths(checkpoint)
    source_hash = digest(checkpoint)
    if source_hash != metadata.get("model_sha256"):
        raise ValueError("Model checksum mismatch; refusing NPU conversion")
    prep = metadata["preprocessing"]
    shape = (round(float(prep["window_seconds"]) * int(prep["sample_rate_hz"])), 52)
    identity = {
        "cache_version": CACHE_VERSION,
        "model_sha256": source_hash,
        "input_shape": [1, 1, *shape],
        "classes": classes,
        "quantization": "Quark XINT8",
    }
    paths["lock"].touch(exist_ok=True)
    with paths["lock"].open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if (
                paths["manifest"].is_file()
                and paths["onnx"].is_file()
                and paths["int8"].is_file()
            ):
                try:
                    cached = json.loads(paths["manifest"].read_text())
                    matches = all(
                        cached.get(key) == value for key, value in identity.items()
                    )
                    intact = cached.get("onnx_sha256") == digest(
                        paths["onnx"]
                    ) and cached.get("int8_sha256") == digest(paths["int8"])
                    if matches and intact:
                        return paths
                except (ValueError, OSError):
                    pass
            import torch
            from quark.onnx import ModelQuantizer
            from quark.onnx.quantization.config import Config, get_default_config

            token = uuid.uuid4().hex
            temporary_onnx = paths["onnx"].with_name(
                f".{paths['onnx'].name}.{token}.tmp.onnx"
            )
            temporary_int8 = paths["int8"].with_name(
                f".{paths['int8'].name}.{token}.tmp.onnx"
            )
            temporary_manifest = paths["manifest"].with_name(
                f".{paths['manifest'].name}.{token}.tmp"
            )
            try:
                model = _load_model(checkpoint, len(classes))
                sample = torch.zeros(identity["input_shape"], dtype=torch.float32)
                torch.onnx.export(
                    model,
                    sample,
                    temporary_onnx,
                    export_params=True,
                    opset_version=17,
                    input_names=["input"],
                    output_names=["logits"],
                    dynamo=False,
                )
                windows = calibration_windows(checkpoint, classes, shape)
                config = Config(global_quant_config=get_default_config("XINT8"))
                ModelQuantizer(config).quantize_model(
                    model_input=str(temporary_onnx),
                    model_output=str(temporary_int8),
                    calibration_data_reader=CalibrationDataReader(windows),
                )
                manifest = {
                    **identity,
                    "calibration_samples": len(windows),
                    "onnx_sha256": digest(temporary_onnx),
                    "int8_sha256": digest(temporary_int8),
                }
                temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
                os.replace(temporary_onnx, paths["onnx"])
                os.replace(temporary_int8, paths["int8"])
                os.replace(temporary_manifest, paths["manifest"])
                shutil.rmtree(paths["vaip_cache"], ignore_errors=True)
            finally:
                for temporary in (temporary_onnx, temporary_int8, temporary_manifest):
                    temporary.unlink(missing_ok=True)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return paths


class NpuModel:
    """A fixed-batch XINT8 Vitis AI session for one immutable model run."""

    def __init__(self, metadata: dict, checkpoint: Path, classes: list[str]):
        import onnxruntime as ort

        if "VitisAIExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("Ryzen AI VitisAIExecutionProvider is unavailable")
        self.paths = prepare_int8_model(metadata, checkpoint, classes)
        self.classes = classes
        self.paths["vaip_cache"].mkdir(exist_ok=True)
        options = ort.SessionOptions()
        options.log_severity_level = 2
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        provider_options = [
            {
                "cache_dir": str(self.paths["vaip_cache"]),
                "cache_key": metadata["model_sha256"][:24],
                "enable_cache_file_io_in_mem": "0",
            }
        ]
        self.session = ort.InferenceSession(
            str(self.paths["int8"]),
            sess_options=options,
            providers=["VitisAIExecutionProvider"],
            provider_options=provider_options,
        )
        self.input_name = self.session.get_inputs()[0].name
        expected = self.session.get_inputs()[0].shape
        self.input_shape = tuple(expected[1:])
        if expected[0] != 1:
            raise RuntimeError(
                f"NPU model must have fixed batch size 1, found {expected}"
            )
        # Compile/warm the graph before live hardware is acquired.
        self.infer(np.zeros(expected, dtype=np.float32))

    def infer(self, batch: np.ndarray) -> tuple[np.ndarray, float]:
        batch = np.asarray(batch, dtype=np.float32)
        if (
            batch.ndim != 4
            or not 1 <= batch.shape[0] <= 8
            or batch.shape[1:] != self.input_shape
        ):
            raise ValueError(f"Invalid NPU input shape: {batch.shape}")
        began = time.monotonic()
        logits = []
        for sample in batch:
            value = self.session.run(None, {self.input_name: sample[None]})[0]
            logits.append(np.asarray(value, dtype=np.float32)[0])
        logits = np.stack(logits)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities, (time.monotonic() - began) * 1000
