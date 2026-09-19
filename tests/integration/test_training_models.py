"""Run these in the training image; collection-only images need no torch."""

import csv
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "csi_model/finetune/session_tools"))
from train_session import interval_split


def test_interval_split_keeps_receivers_and_sessions_together():
    rows = [
        dict(session_id=s, interval_id=i, label=label, receiver=rx, filename=f"{w}.mat")
        for s in ["a", "b"]
        for i, label in [("event1", "Static"), ("event2", "Walking")]
        for rx in ["left", "right"]
        for w in range(3)
    ]
    frame = pd.DataFrame(rows)
    train, val = interval_split(frame)
    key = lambda m: set(zip(m.session_id, m.interval_id))
    assert len(val) > 0
    assert not key(train) & key(val)
    assert set(train.label) == {"Static", "Walking"}
    assert len(train) + len(val) == len(frame)


def test_labeling_uses_vfr_pts_and_breaks_missing_camera_gaps(tmp_path, monkeypatch):
    from pose_labeling import inference

    raw = tmp_path / "session/raw"
    raw.mkdir(parents=True)
    video = raw / "video.mp4"
    video.touch()
    # Two six-second spans separated by a network/camera gap. Nominal frame/FPS
    # would place the second span at 6s, but its capture PTS actually starts at 20s.
    pts = np.r_[np.arange(0, 6, 0.1), np.arange(20, 26, 0.1)]
    rows = [
        dict(
            frame_idx=i,
            video_pts_s=float(p),
            capture_timestamp_ns=10**12 + round(p * 1e9),
            host_timestamp_ns=2 * 10**12 + round(i * 0.1 * 1e9),
        )
        for i, p in enumerate(pts)
    ]
    index = raw / "video_frames.parquet"
    pq.write_table(pa.Table.from_pylist(rows), index)

    class Decoded:
        def __init__(self, p):
            self.pts, self.time_base = round(p * 1000), 0.001

        def to_ndarray(self, **kwargs):
            return np.zeros((240, 320, 3), dtype=np.uint8)

    class Video:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def decode(self, **kwargs):
            return iter(Decoded(p) for p in pts)

    class Yolo:
        def __init__(self, *args):
            pass

        def to(self, device):
            assert device == "cuda:0"
            return self

        def predict(self, frame, **kwargs):
            assert kwargs["device"] == "cuda:0"
            return [
                types.SimpleNamespace(
                    keypoints=types.SimpleNamespace(data=torch.ones((1, 17, 3)))
                )
            ]

    class Model:
        def __init__(self, **kwargs):
            pass

        def to(self, *args):
            return self

        def load_state_dict(self, *args):
            pass

        def eval(self):
            pass

    module = types.ModuleType("model.ctrgcn")
    module.Model = Model
    monkeypatch.setitem(sys.modules, "model.ctrgcn", module)
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=Yolo))
    monkeypatch.setattr(inference.av, "open", lambda *args: Video())
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "test")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    csv_path = inference.run_text_inference(video, tmp_path / "labels.csv")
    labels = list(csv.DictReader(csv_path.open()))
    assert len(labels) == 2
    assert float(labels[0]["end_time"]) < 6000
    assert float(labels[1]["start_time"]) >= 20000
    assert int(labels[1]["start_host_timestamp_ns"]) >= 10**12 + 20 * 10**9


def test_auto_pipeline_stage_order_and_preserved_previous_model(tmp_path, monkeypatch):
    from apps.training_service.app import pipeline

    session = tmp_path / "session"
    run = session / "train/runs/new"
    run.mkdir(parents=True)
    (session / "train/finetuned_resnet18.pth").write_bytes(b"previous model")
    events = []
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def label(video, output):
        events.append("label")
        output.write_text("start_time,end_time,label\n0,3000,Walking\n")
        output.with_suffix(".json").write_text("{}")

    monkeypatch.setitem(
        sys.modules,
        "pose_labeling.inference",
        types.SimpleNamespace(run_text_inference=label),
    )

    def preprocess(*args, **kwargs):
        events.append("preprocess")
        raise ValueError("Not enough usable data")

    monkeypatch.setattr(pipeline, "build_dataset", preprocess)
    from apps.common.schemas import TrainRequest

    with pytest.raises(ValueError, match="Not enough"):
        pipeline.run(session, run, TrainRequest(action="auto").model_dump())
    assert events == ["label", "preprocess"]
    assert (session / "train/action_results.csv").is_file()
    assert (session / "train/finetuned_resnet18.pth").read_bytes() == b"previous model"


def test_deploy_loads_trained_head_and_uses_training_normalization(
    tmp_path, monkeypatch
):
    import hashlib
    import scipy.io
    from apps.training_service.app.deploy import Predictor
    from finetune import normalize_amplitude, load_mat_amplitude

    sys.path.insert(0, str(ROOT / "csi_model/Model Code"))
    from ESP_Fi_model import ESP_Fi_ResNet18

    # Exercise checkpoint loading/forward without taking the GPU from an active job.
    model = ESP_Fi_ResNet18(num_classes=2)
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.copy_(torch.tensor([0.0, 2.0]))
    checkpoint = tmp_path / "finetuned.pth"
    torch.save(model.state_dict(), checkpoint)
    metadata = dict(
        model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        preprocessing=dict(window_seconds=2, sample_rate_hz=100),
    )
    original_tensor_to, original_module_to = torch.Tensor.to, torch.nn.Module.to
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.Tensor,
        "to",
        lambda self, *a, **k: original_tensor_to(
            self, *("cpu" if x == "cuda:0" else x for x in a), **k
        ),
    )
    monkeypatch.setattr(
        torch.nn.Module,
        "to",
        lambda self, *a, **k: original_module_to(
            self, *("cpu" if x == "cuda:0" else x for x in a), **k
        ),
    )
    amplitudes = np.random.default_rng(42).uniform(0, 100, (200, 52)).astype(np.float32)
    scipy.io.savemat(tmp_path / "window.mat", {"CSIamp": amplitudes})
    np.testing.assert_array_equal(
        normalize_amplitude(amplitudes), load_mat_amplitude(tmp_path / "window.mat")
    )
    predictor = Predictor(metadata, checkpoint, ["Static", "Walking"])
    result = predictor.predict(amplitudes)
    assert result["label"] == "Walking"
    assert result["confidence"] == pytest.approx(
        float(torch.softmax(torch.tensor([0.0, 2.0]), dim=0)[1])
    )
    predictor.close()
    with pytest.raises(ValueError, match="checksum"):
        Predictor(
            {**metadata, "model_sha256": "wrong"}, checkpoint, ["Static", "Walking"]
        )
