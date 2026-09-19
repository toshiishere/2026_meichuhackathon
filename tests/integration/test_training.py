"""Training orchestration and preprocessing checks; no GPU required here."""

import csv
import hashlib
import io
import json
from pathlib import Path
import threading
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zstandard
from fastapi.testclient import TestClient

from apps.common.schemas import TrainRequest
from apps.common.session_lock import session_lock
from apps.common.storage import atomic_json
from apps.training_service.app import main
from apps.training_service.app.preprocess import build_dataset, decode_packets
from apps.training_service.app.timeline import Timeline


def make_session(root):
    path = root / "sessions/train_test"
    (path / "raw").mkdir(parents=True)
    (path / "train").mkdir()
    atomic_json(
        path / "metadata.json",
        dict(schema_version="1.0", session_id="train_test", status="complete"),
    )
    (path / "raw/video.mp4").write_bytes(b"video placeholder for API validation")
    frames = [
        dict(
            frame_idx=i,
            video_pts_s=i / 20,
            capture_timestamp_ns=10_000_000_000 + i * 50_000_000,
            host_timestamp_ns=10_000_000_000
            + i * 50_000_000
            + (5_000_000_000 if i >= 60 else 0),
        )
        for i in range(161)
    ]
    pq.write_table(pa.Table.from_pylist(frames), path / "raw/video_frames.parquet")
    text = io.StringIO()
    writer = csv.DictWriter(
        text,
        fieldnames=[
            "host_timestamp_ns",
            "bandwidth",
            "sig_mode",
            "stbc",
            "len",
            "first_word",
            "data",
        ],
    )
    writer.writeheader()
    for i in range(801):
        writer.writerow(
            dict(
                host_timestamp_ns=10_000_000_000 + i * 10_000_000,
                bandwidth=0,
                sig_mode=1,
                stbc=0,
                len=256,
                first_word=0,
                data=json.dumps([(j + i) % 60 - 30 for j in range(256)]),
            )
        )
    (path / "raw/csi_rx.csv.zst").write_bytes(
        zstandard.ZstdCompressor().compress(text.getvalue().encode())
    )
    (path / "train/action_results.csv").write_text(
        "start_time,end_time,label\n0,4000,Walking\n4000,8000,Static\n"
    )
    return path


def test_preprocessing_uses_capture_clock_and_preserves_raw(tmp_path):
    session = make_session(tmp_path)
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (session / "raw").iterdir()
    }
    summary = build_dataset(
        session, session / "train/action_results.csv", session / "train/dataset"
    )
    assert summary["windows"] == 6
    assert summary["class_counts"] == {"Walking": 3, "Static": 3}
    assert summary["clock_column"] == "capture_timestamp_ns"
    rows = list(csv.DictReader((session / "train/dataset/manifest.csv").open()))
    assert int(rows[3]["capture_start_ns"]) == 14_000_000_000  # Not receipt time 19s.
    import scipy.io

    array = scipy.io.loadmat(session / "train/dataset/Static" / rows[3]["filename"])[
        "CSIamp"
    ]
    assert array.shape == (200, 52) and np.isfinite(array).all()
    assert before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (session / "raw").iterdir()
    }
    assert not (session / "raw/csi_rx.csv").exists()


def test_preprocessing_rejects_bad_labels_and_large_gaps(tmp_path):
    session = make_session(tmp_path)
    labels = session / "train/action_results.csv"
    labels.write_text("start_time,end_time,label\n0,4000,Walking\n3000,5000,Static\n")
    with pytest.raises(ValueError, match="overlap"):
        build_dataset(session, labels, session / "train/bad")
    labels.write_text("start_time,end_time,label\n0,9000,Walking\n")
    with pytest.raises(ValueError, match="Invalid label"):
        build_dataset(session, labels, session / "train/bad2")
    labels.write_text("start_time,end_time,label\n0,8000,Walking\n")
    index = session / "raw/video_frames.parquet"
    rows = pq.read_table(index).to_pylist()
    rows = [r for r in rows if not 1 < r["video_pts_s"] < 7]
    for i, r in enumerate(rows):
        r["frame_idx"] = i
    pq.write_table(pa.Table.from_pylist(rows), index)
    with pytest.raises(ValueError, match="No usable CSI windows"):
        build_dataset(session, labels, session / "train/gaps")


def test_timeline_legacy_and_unsupported_binary(tmp_path):
    session = make_session(tmp_path)
    index = session / "raw/video_frames.parquet"
    rows = pq.read_table(index).drop_columns(["capture_timestamp_ns"])
    pq.write_table(rows, index)
    assert Timeline(index).clock_column == "host_timestamp_ns"
    wire = tmp_path / "wire.csv"
    wire.write_text('host_timestamp_ns,data\n,"[1,2]"\n')
    with pytest.raises(ValueError, match="standalone wire binary"):
        decode_packets(wire)


def test_training_api_locking_logs_cancellation_and_restart(tmp_path, monkeypatch):
    path = make_session(tmp_path)
    monkeypatch.setattr(main, "DATA", tmp_path)
    monkeypatch.setattr(
        main, "gpu_info", lambda: dict(ready=True, rocm="test", device="test")
    )
    finished = threading.Event()

    class Process:
        pid = 987654

        def __init__(self, command, stdout, **kwargs):
            assert command[:4] == [
                main.sys.executable,
                "-u",
                "-m",
                "apps.training_service.app.pipeline",
            ]
            assert kwargs["start_new_session"]
            stdout.write("Preprocessing session\n")

        def poll(self):
            return -15 if finished.is_set() else None

        def wait(self, timeout=None):
            finished.wait(1)
            return -15

    monkeypatch.setattr(main.subprocess, "Popen", Process)
    monkeypatch.setattr(main.os, "killpg", lambda *args: finished.set())
    with TestClient(main.app) as client:
        with session_lock(tmp_path, "train_test"):
            assert (
                client.post(
                    "/sessions/train_test/start", json={"action": "auto"}
                ).status_code
                == 409
            )
        result = client.post("/sessions/train_test/start", json={"action": "auto"})
        assert result.status_code == 200, result.text
        jid = result.json()["id"]
        assert (
            client.post(
                "/sessions/train_test/start", json={"action": "auto"}
            ).status_code
            == 409
        )
        with pytest.raises(RuntimeError, match="busy"):
            with session_lock(tmp_path, "train_test"):
                pass
        assert client.post("/jobs/" + jid + "/cancel").status_code == 200
        for _ in range(100):
            job = client.get("/sessions/train_test").json()["jobs"][0]
            if job["status"] == "cancelled":
                break
            time.sleep(0.02)
        assert job["status"] == "cancelled"
        assert "Preprocessing" in client.get("/jobs/" + jid + "/logs").json()["logs"]
        with session_lock(tmp_path, "train_test"):
            pass
        assert (
            path / "raw/video.mp4"
        ).read_bytes() == b"video placeholder for API validation"
        main.manager.db.put(
            "jobs",
            "interrupted",
            dict(id="interrupted", session_id="train_test", status="running"),
        )
    with TestClient(main.app):
        assert main.manager.db.get("jobs", "interrupted")["status"] == "failed"


def test_training_requires_completed_session_and_gpu(tmp_path, monkeypatch):
    session = make_session(tmp_path)
    monkeypatch.setattr(main, "DATA", tmp_path)
    monkeypatch.setattr(
        main, "gpu_info", lambda: dict(ready=False, error="No ROCm device")
    )
    with TestClient(main.app) as client:
        assert (
            client.post(
                "/sessions/train_test/start", json={"action": "auto"}
            ).status_code
            == 503
        )
        atomic_json(session / "metadata.json", dict(status="recording"))
        assert (
            client.post(
                "/sessions/train_test/start", json={"action": "label"}
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/sessions/train_test/start", json={"action": "shell"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/sessions/train_test/start",
                json={"action": "auto", "epochs_frozen": 0, "epochs_finetune": 0},
            ).status_code
            == 422
        )
        assert client.get("/sessions/invalid!").status_code == 400
        assert (
            client.post("/sessions/missing/start", json={"action": "auto"}).status_code
            == 404
        )
