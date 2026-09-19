"""Deployment preprocessing parity, lifecycle, input validation and resource ownership."""

import csv
import hashlib
import json
import threading
import time

import numpy as np
import pytest
import scipy.io
from fastapi import HTTPException
from fastapi.testclient import TestClient

from apps.common.schemas import DeployRequest, DeployCaptureRequest
from apps.common.session_lock import session_lock
from apps.common.storage import atomic_json
from apps.training_service.app import deploy, main
from apps.training_service.app.preprocess import build_dataset, packet_rows
from test_training import make_session


def model_fixture(root):
    session = make_session(root)
    run = session / "train/runs" / ("a" * 32)
    run.mkdir(parents=True)
    checkpoint = run / "finetuned_resnet18.pth"
    checkpoint.write_bytes(b"test model")
    atomic_json(
        run / "classes.json",
        dict(
            class_names=["Static", "Walking"], class_to_idx={"Static": 0, "Walking": 1}
        ),
    )
    atomic_json(
        session / "train/model.json",
        dict(
            architecture="ESP_Fi_ResNet18",
            run_id=run.name,
            model_path=str(checkpoint.relative_to(session)),
            model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            preprocessing=dict(
                window_seconds=2,
                sample_rate_hz=100,
                overlap=0.5,
                normalization="per-window global z-score; applied by training loader",
            ),
        ),
    )
    return session


class FakePredictor:
    def __init__(self, *args):
        pass

    def predict(self, amp):
        assert amp.shape == (200, 52)
        return dict(
            label="Walking",
            confidence=0.8,
            scores={"Static": 0.2, "Walking": 0.8},
            inference_ms=1,
        )

    def close(self):
        pass


def wait_for(fn, timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if fn():
            return
        time.sleep(0.02)
    assert fn()


def test_live_window_matches_training_mat_and_rejects_gaps(tmp_path):
    session = make_session(tmp_path)
    build_dataset(
        session, session / "train/action_results.csv", session / "train/dataset"
    )
    buffer = deploy.WindowBuffer(2, 100)
    rows = list(packet_rows(session / "raw/csi_rx.csv.zst"))
    for row in rows[:201]:
        buffer.add(row)
    target = scipy.io.loadmat(session / "train/dataset/Walking/rx-000000.mat")["CSIamp"]
    np.testing.assert_array_equal(buffer.window(), target)
    assert not buffer.add(rows[200])
    assert not buffer.add({**rows[201], "first_word": 1})
    assert not buffer.add({**rows[201], "len": 128})
    for row in rows[250:301]:
        buffer.add(row)
    assert buffer.window() is None  # 500ms gap must not become a prediction.
    assert buffer.rejected == {
        "duplicate_or_out_of_order": 1,
        "invalid_first_word": 1,
        "unsupported_layout": 1,
    }


def test_replay_lifecycle_model_pinning_and_session_lock(tmp_path, monkeypatch):
    session = model_fixture(tmp_path)
    monkeypatch.setattr(deploy, "Predictor", FakePredictor)
    gpu_guard = threading.Lock()
    manager = deploy.Deployment(tmp_path, gpu_guard, main.session_path)
    options = DeployRequest(
        model_session_id=session.name,
        source="replay",
        replay_session_id=session.name,
        replay_receiver="rx",
        replay_speed=4,
    )
    catalog = manager.catalog()
    assert catalog["models"][0]["classes"] == ["Static", "Walking"]
    assert catalog["sources"][0]["receivers"] == ["rx"]
    manager.start(options)
    with pytest.raises(HTTPException) as exc:
        manager.start(options)
    assert exc.value.status_code == 409
    with pytest.raises(RuntimeError):
        with session_lock(tmp_path, session.name):
            pass
    wait_for(lambda: manager.snapshot()["status"] == "completed")
    state = manager.snapshot()
    assert state["prediction"]["label"] == "Walking" and len(state["history"]) >= 2
    assert state["accepted"] == 801 and state["source_elapsed_s"] == 8
    assert state["model_run_id"] == "a" * 32
    assert state["video_available"]
    assert state["video_time_s"] == pytest.approx(8, abs=0.15)
    assert state["source_timestamp_ns"] == 18_000_000_000
    with session_lock(tmp_path, session.name):
        pass
    assert not gpu_guard.locked()
    manager.start(options)
    manager.stop()
    wait_for(lambda: manager.snapshot()["status"] == "stopped")
    assert manager.snapshot()["prediction"] is None and not gpu_guard.locked()


def test_bad_artifacts_and_failed_gpu_release_leases(tmp_path, monkeypatch):
    session = model_fixture(tmp_path)
    guard = threading.Lock()
    manager = deploy.Deployment(tmp_path, guard, main.session_path)
    options = DeployRequest(
        model_session_id=session.name,
        source="replay",
        replay_session_id=session.name,
        replay_receiver="rx",
    )

    def broken(*args):
        raise RuntimeError("GPU unavailable")

    monkeypatch.setattr(deploy, "Predictor", broken)
    manager.start(options)
    wait_for(lambda: manager.snapshot()["status"] == "failed")
    assert manager.snapshot()["error"] == "GPU unavailable" and not guard.locked()
    pointer = session / "train/model.json"
    data = json.loads(pointer.read_text())
    data["model_path"] = "../../outside.pth"
    atomic_json(pointer, data)
    with pytest.raises(ValueError, match="immutable"):
        manager.start(options)
    assert not guard.locked()
    assert manager.catalog()["errors"]
    with pytest.raises(ValueError):
        DeployRequest(model_session_id="../escape", source="live")


def test_capture_excludes_recording_and_releases_on_disconnect(tmp_path, monkeypatch):
    from apps.hardware_service.app import deploy as capture
    from apps.hardware_service.app.jobs import Jobs, BusyError

    jobs = Jobs(tmp_path)
    handle = capture.DeployCapture(jobs)
    options = DeployCaptureRequest(
        receiver=dict(logical_name="rx", port="synthetic://rx0")
    )
    result = handle.start(options)
    try:
        with pytest.raises(BusyError):
            jobs.acquire()
        wait_for(lambda: len(handle.rows) > 2)
        batch = handle.batch(result["capture_id"])
        assert batch["active"] and len(batch["rows"]) > 0
        assert "host_timestamp_ns" in batch["rows"][0]
        with pytest.raises(HTTPException):
            handle.batch("wrong-token")
        handle.last_poll = time.monotonic() - 16
        wait_for(lambda: not handle.active)
        assert "disconnected" in handle.error
    finally:
        handle.close()
    jobs.acquire()
    jobs.lease.release()


def test_deployment_api_validation_and_status(tmp_path, monkeypatch):
    session = model_fixture(tmp_path)
    monkeypatch.setattr(main, "DATA", tmp_path)
    monkeypatch.setattr(deploy, "Predictor", FakePredictor)
    with TestClient(main.app) as client:
        assert (
            client.get("/deploy/catalog").json()["models"][0]["session_id"]
            == session.name
        )
        assert (
            client.post(
                "/deploy/start", json=dict(model_session_id=session.name, source="live")
            ).status_code
            == 422
        )
        result = client.post(
            "/deploy/start",
            json=dict(
                model_session_id=session.name,
                source="replay",
                replay_session_id=session.name,
                replay_receiver="rx",
                replay_speed=4,
            ),
        )
        assert result.status_code == 200, result.text
        assert client.get("/deploy/status").json()["options"]["source"] == "replay"
        assert client.post("/deploy/stop").status_code == 200


def test_binary_serial_capture_decodes_and_feeds_the_training_layout(
    tmp_path, monkeypatch
):
    import struct
    import zlib
    from apps.hardware_service.app import deploy as capture
    from apps.hardware_service.app.jobs import Jobs
    from apps.hardware_service.app.csi_wire import HEADER, META, PHY_FIELDS
    from tests.test_csi_binary import binary_frame
    from apps.training_service.app.preprocess import packet_amplitude

    packet = binary_frame()
    metadata = list(META.unpack_from(packet, HEADER.size))
    metadata[13 + PHY_FIELDS.index("bandwidth")] = 0
    metadata[13 + PHY_FIELDS.index("first_word")] = 0
    packet = (
        packet[: HEADER.size]
        + META.pack(*metadata)
        + packet[HEADER.size + META.size : -4]
    )
    packet += struct.pack("<I", zlib.crc32(packet[4:]))

    class BinarySource:
        closed = False

        def __init__(self, *args):
            pass

        def read(self, stop):
            if stop.wait(0.01):
                return None
            return packet, time.monotonic_ns()

        def close(self):
            self.closed = True

    monkeypatch.setattr(capture, "SerialSource", BinarySource)
    handle = capture.DeployCapture(Jobs(tmp_path))
    token = handle.start(
        DeployCaptureRequest(receiver=dict(logical_name="rx", port="synthetic://rx0"))
    )["capture_id"]
    try:
        wait_for(lambda: len(handle.rows) > 1)
        data = handle.batch(token)
        assert data["rows"][0]["firmware_layout"] == "binary_v1"
        amp = packet_amplitude(data["rows"][0])
        assert amp.shape == (52,) and np.isfinite(amp).all()
    finally:
        handle.close()


def test_replay_accepts_another_session_and_cancellation_releases_both(
    tmp_path, monkeypatch
):
    import shutil

    source = model_fixture(tmp_path)
    other = tmp_path / "sessions/other"
    shutil.copytree(source, other)
    monkeypatch.setattr(deploy, "Predictor", FakePredictor)
    manager = deploy.Deployment(tmp_path, threading.Lock(), main.session_path)
    manager.start(
        DeployRequest(
            model_session_id=source.name,
            source="replay",
            replay_session_id="other",
            replay_receiver="rx",
        )
    )
    try:
        for sid in (source.name, "other"):
            with pytest.raises(RuntimeError):
                with session_lock(tmp_path, sid):
                    pass
    finally:
        manager.stop()
        manager.close()
    for sid in (source.name, "other"):
        with session_lock(tmp_path, sid):
            pass


def test_live_worker_clears_stale_pose_camera_failure_is_independent(
    tmp_path, monkeypatch
):
    import httpx

    session = model_fixture(tmp_path)
    packets = list(packet_rows(session / "raw/csi_rx.csv.zst"))[:251]
    requests = []

    def transport(request):
        requests.append(request.url.path)
        if request.url.path == "/deploy/capture":
            return httpx.Response(200, json={"capture_id": "b" * 32})
        if request.url.path.endswith("/packets"):
            rows = packets[:]
            packets.clear()
            return httpx.Response(
                200,
                json=dict(
                    rows=rows,
                    active=True,
                    error=None,
                    dropped=0,
                    malformed=0,
                    camera_error="Camera disconnected",
                ),
            )
        return httpx.Response(200, json={"stopping": False})

    original = httpx.Client
    monkeypatch.setattr(
        deploy.httpx,
        "Client",
        lambda **kw: original(transport=httpx.MockTransport(transport), **kw),
    )
    monkeypatch.setattr(deploy, "Predictor", FakePredictor)
    manager = deploy.Deployment(tmp_path, threading.Lock(), main.session_path)
    manager.start(
        DeployRequest(
            model_session_id=session.name,
            source="live",
            receiver=dict(logical_name="rx", port="synthetic://rx0"),
            camera=dict(device="synthetic://camera"),
        )
    )
    try:
        wait_for(lambda: manager.snapshot().get("prediction") is not None)
        assert manager.snapshot()["camera_error"] == "Camera disconnected"
        wait_for(lambda: manager.snapshot()["signal"] == "no_data")
        assert manager.snapshot()["prediction"] is None
        assert manager.snapshot()["status"] == "running"
    finally:
        manager.stop()
        manager.close()
    assert requests[-1].endswith("/stop") and not manager.gpu_guard.locked()
    with session_lock(tmp_path, session.name):
        pass
