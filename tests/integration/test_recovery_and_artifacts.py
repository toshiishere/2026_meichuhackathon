import json
import threading
import time
from types import SimpleNamespace

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps.common.session_lock import session_lock
from apps.common.storage import atomic_json
from apps.hardware_service.app import main as hardware
from apps.hardware_service.app.recovery import recover_session
from apps.training_service.app import main as training
from test_deploy import model_fixture
from test_training import make_session


def incomplete(root):
    session = make_session(root)
    with av.open(str(session / "raw/video.mp4"), "w") as container:
        stream = container.add_stream("libx264", rate=20)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.options = {"bf": "0"}
        for i in range(161):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((64, 64, 3), dtype=np.uint8), format="rgb24"
            )
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    atomic_json(
        session / "metadata.json",
        dict(
            schema_version="1.0",
            session_id=session.name,
            status="incomplete",
            quality="incomplete",
            errors=["Camera stalled"],
            configuration=dict(receivers=[dict(logical_name="rx")]),
        ),
    )
    for file in (session / "raw").iterdir():
        file.rename(file.with_name(f".{file.name}.tmp"))
    return session


def test_recovery_renames_without_changing_content_and_keeps_provenance(tmp_path):
    session = incomplete(tmp_path)
    before = {p.name: p.read_bytes() for p in (session / "raw").iterdir()}
    result = recover_session(tmp_path, session.name, lambda _: None, threading.Event())
    assert result["renamed"] == 3
    for name, content in before.items():
        assert (session / "raw" / name[1:-4]).read_bytes() == content
    metadata = json.loads((session / "metadata.json").read_text())
    assert metadata["status"] == "complete" and metadata["recovered"]
    assert metadata["quality"] == "recovered" and metadata["errors"] == [
        "Camera stalled"
    ]
    assert metadata["recovered_at"] and (tmp_path / "manifest.parquet").is_file()
    with pytest.raises(ValueError, match="Only incomplete"):
        recover_session(tmp_path, session.name, print, threading.Event())


@pytest.mark.parametrize(
    "failure", ["missing", "conflict", "symlink", "corrupt", "cancelled"]
)
def test_recovery_validates_before_any_rename(tmp_path, failure):
    session = incomplete(tmp_path)
    file = session / "raw/.video.mp4.tmp"
    stop = threading.Event()
    if failure == "missing":
        file.unlink()
    elif failure == "conflict":
        (session / "raw/video.mp4").write_bytes(b"existing final")
    elif failure == "symlink":
        file.unlink()
        file.symlink_to(session / "metadata.json")
    elif failure == "corrupt":
        file.write_bytes(b"not a video")
    else:
        stop.set()
    with pytest.raises(Exception):
        recover_session(tmp_path, session.name, print, stop)
    assert (session / "raw/.csi_rx.csv.zst.tmp").exists()
    assert not (session / "raw/csi_rx.csv.zst").exists()
    assert json.loads((session / "metadata.json").read_text())["status"] == "incomplete"


def test_recovery_job_excludes_busy_and_active_sessions(tmp_path, monkeypatch):
    session = incomplete(tmp_path)
    monkeypatch.setattr(hardware, "DATA", tmp_path)
    monkeypatch.setattr(hardware, "recorder", None)
    with TestClient(hardware.app) as client:
        hardware.jobs.acquire()
        try:
            assert client.post(f"/sessions/{session.name}/recover").status_code == 409
        finally:
            hardware.jobs.lease.release()
        monkeypatch.setattr(
            hardware,
            "recorder",
            SimpleNamespace(
                config=SimpleNamespace(session_id=session.name),
                state="incomplete",
                threads=[SimpleNamespace(is_alive=lambda: True)],
            ),
        )
        job = client.post(f"/sessions/{session.name}/recover").json()
        for _ in range(100):
            job = client.get(f"/jobs/{job['id']}").json()
            if job["status"] == "failed":
                break
            time.sleep(0.02)
        assert job["status"] == "failed" and "active acquisition" in job["error"]
        assert (session / "raw/.video.mp4.tmp").exists()
        monkeypatch.setattr(hardware, "recorder", None)
        job = client.post(f"/sessions/{session.name}/recover").json()
        for _ in range(100):
            job = client.get(f"/jobs/{job['id']}").json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.02)
        assert job["status"] == "completed", job


def test_delete_labels_models_separately_and_respect_locks(tmp_path, monkeypatch):
    session = model_fixture(tmp_path)
    train = session / "train"
    run = train / "runs" / ("a" * 32)
    (run / "action_results.csv").write_text("old labels")
    (run / "labels_used.csv").write_text("provenance")
    (run / "job.log").write_text("log")
    (train / "finetuned_resnet18.pth").write_bytes(b"published model")
    monkeypatch.setattr(training, "DATA", tmp_path)
    with TestClient(training.app) as client:
        endpoint = f"/sessions/{session.name}/remove"
        training.manager.guard.acquire()
        try:
            assert client.post(endpoint + "/labels").status_code == 409
        finally:
            training.manager.guard.release()
        with session_lock(tmp_path, session.name):
            assert client.post(endpoint + "/model").status_code == 409
        assert client.post(endpoint + "/labels").status_code == 200
        state = client.get(f"/sessions/{session.name}").json()
        assert not state["labels_ready"] and state["model"]
        assert not (run / "action_results.csv").exists()
        assert (run / "labels_used.csv").read_text() == "provenance"
        assert client.post(endpoint + "/model").status_code == 200
        assert client.get(f"/sessions/{session.name}").json()["model"] is None
        assert not (run / "finetuned_resnet18.pth").exists()
        assert not (train / "finetuned_resnet18.pth").exists()
        assert (run / "job.log").is_file() and (session / "raw/video.mp4").is_file()
        assert client.post(endpoint + "/raw").status_code == 404


def test_deletion_does_not_follow_symlinks(tmp_path, monkeypatch):
    session = model_fixture(tmp_path)
    outside = tmp_path / "outside.csv"
    outside.write_text("keep me")
    labels = session / "train/action_results.csv"
    labels.unlink()
    labels.symlink_to(outside)
    monkeypatch.setattr(training, "DATA", tmp_path)
    with TestClient(training.app) as client:
        assert client.post(f"/sessions/{session.name}/remove/labels").status_code == 400
    assert outside.read_text() == "keep me"
