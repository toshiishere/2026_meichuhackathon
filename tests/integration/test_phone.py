import threading
import time
import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from apps.hardware_service.app import main, phone, sources
from apps.hardware_service.app.recorder import Recorder
from apps.common.schemas import CameraConfig


def frame(seq=0, capture=10.0, skipped=0, width=320, height=240):
    ok, encoded = cv2.imencode(".jpg", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return phone.HEADER.pack(b"CSI1", seq, capture, skipped) + encoded.tobytes()


@pytest.fixture
def hub(monkeypatch):
    value = phone.PhoneHub()
    monkeypatch.setattr(phone, "hub", value)
    monkeypatch.setattr(main, "hub", value)
    monkeypatch.setattr(sources, "hub", value)
    return value


def test_pairing_expiry_reuse_and_frame_validation(hub):
    settings = dict(name="Phone", width=320, height=240, fps=20)
    pairing = hub.pair(settings)
    with pytest.raises(ValueError):
        hub.settings(pairing["id"], "wrong token")
    hub.connect(pairing["id"], pairing["token"], [1920, 1080])
    with pytest.raises(ValueError):
        hub.connect(pairing["id"], pairing["token"], [320, 240])
    assert hub.accept_frame(pairing["id"], frame(), 123456)["type"] == "ack"
    with pytest.raises(ValueError):
        hub.accept_frame(pairing["id"], frame(), 123457)
    with pytest.raises(ValueError):
        hub.accept_frame(pairing["id"], frame(seq=1, capture=11, width=640), 123457)
    with pytest.raises(ValueError):
        hub.accept_frame(pairing["id"], frame(seq=1, capture=float("nan")), 123457)
    with pytest.raises(ValueError):
        hub.accept_frame(pairing["id"], b"x" * (phone.MAX_FRAME_BYTES + 1), 123457)
    hub.disconnect(pairing["id"])
    with pytest.raises(ValueError):
        hub.settings(pairing["id"], pairing["token"])
    pending = hub.pair(settings)
    hub.phones[pending["id"]].expires = 0
    with pytest.raises(ValueError):
        hub.settings(pending["id"], pending["token"])


def test_phone_source_loss_disconnect_and_resolution(hub, monkeypatch):
    pairing = hub.pair(dict(name="Phone", width=320, height=240, fps=20))
    hub.connect(pairing["id"], pairing["token"], [320, 240])
    config = CameraConfig(
        device="phone://" + pairing["id"], width=320, height=240, fps=20
    )
    with pytest.raises(ValueError):
        sources.CameraSource(config.model_copy(update={"fps": 30}))
    monkeypatch.setitem(phone.DEFAULTS, "camera_queue_size", 1)
    source = sources.CameraSource(config)
    hub.accept_frame(pairing["id"], frame(), 100)
    hub.accept_frame(pairing["id"], frame(seq=1, capture=11, skipped=2), 200)
    assert source.transport_stats == {"phone_queue_drops": 1, "phone_skipped_frames": 2}
    image, stamp = source.read(threading.Event())
    assert stamp == 100 and image.shape == (240, 320, 3)
    assert source.frame_metadata["phone_capture_timestamp_ms"] == 10
    hub.disconnect(pairing["id"])
    with pytest.raises(RuntimeError, match="disconnected"):
        source.read(threading.Event())
    source.close()
    source.close()


def test_phone_frames_record_with_csi_and_separate_clocks(config, tmp_path, hub):
    pairing = hub.pair(dict(name="Phone", width=320, height=240, fps=20))
    hub.connect(pairing["id"], pairing["token"], [640, 480])
    config.camera = CameraConfig(
        device="phone://" + pairing["id"], width=320, height=240, fps=20
    )
    config.duration_seconds = 1.2
    stop = threading.Event()

    def produce():
        seq = 0
        while not stop.wait(0.05):
            hub.accept_frame(
                pairing["id"],
                frame(seq=seq, capture=1234 + seq * 50),
                time.monotonic_ns(),
            )
            seq += 1

    thread = threading.Thread(target=produce)
    thread.start()
    try:
        result = Recorder(config, tmp_path).run()
    finally:
        stop.set()
        thread.join()
    assert result["status"] == "complete", result["errors"]
    assert "websocket arrival" in result["clock"]["camera_timestamp"]
    rows = pq.read_table(
        tmp_path / "sessions/test_session/raw/video_frames.parquet"
    ).to_pylist()
    assert len(rows) >= 20
    assert all(
        r["schema_version"] == "1.2" and r["phone_frame_idx"] is not None for r in rows
    )
    assert all(r["host_timestamp_ns"] >= result["start_timestamp_ns"] for r in rows)
    assert all(r["phone_capture_timestamp_ms"] < 10000 for r in rows)
    assert all(r["host_timestamp_ns"] > 10**9 for r in rows)


def test_websocket_upload_and_wrong_token_cannot_disconnect_active_phone(
    tmp_path, monkeypatch, hub
):
    monkeypatch.setattr(main, "DATA", tmp_path)
    monkeypatch.setattr(main, "recorder", None)
    with TestClient(main.app) as client:
        pairing = client.post(
            "/phone/pair",
            json={"name": "Phone", "width": 320, "height": 240, "fps": 20},
        ).json()
        with client.websocket_connect("/phone/stream") as ws:
            ws.send_json({"id": pairing["id"], "token": pairing["token"]})
            assert ws.receive_json()["type"] == "settings"
            ws.send_json({"type": "ready", "native_resolution": [320, 240]})
            assert ws.receive_json()["type"] == "ready"
            ws.send_bytes(frame())
            assert ws.receive_json() == {"type": "ack", "frame_idx": 0}
            with client.websocket_connect("/phone/stream") as wrong:
                wrong.send_json({"id": pairing["id"], "token": "invalid"})
                assert wrong.receive_json()["type"] == "error"
            assert hub.phones[pairing["id"]].connected
            cameras = client.get("/cameras").json()
            assert any(c["device"] == "phone://" + pairing["id"] for c in cameras)
        assert not hub.phones[pairing["id"]].connected


def connect_resumable(hub, pairing):
    pid = pairing["id"]
    result = hub.connect_resumable(pid, pairing, [320, 240])
    origin = time.monotonic_ns()
    hub.calibrate(pid, result["connection_id"], origin, 0, 1)
    return result, origin


def test_resume_replays_are_idempotent_and_old_socket_cannot_disconnect(hub):
    pairing = hub.pair(dict(name="Phone", width=320, height=240, fps=20))
    ready, origin = connect_resumable(hub, pairing)
    pid, lease = pairing["id"], ready["connection_id"]
    source = sources.CameraSource(
        CameraConfig(device=f"phone://{pid}", width=320, height=240, fps=20)
    )
    source.enable_recording()
    hub.accept_frame(pid, frame(capture=10), origin + 90_000_000, lease)
    hub.disconnect(pid, lease)
    hub.phones[pid].expires = 0  # The original pairing timeout must not expire resume.
    resumed = hub.connect_resumable(pid, {"resume_token": ready["resume_token"]}, [])
    new_lease = resumed["connection_id"]
    hub.disconnect(pid, lease)
    assert hub.phones[pid].connected
    assert (
        hub.accept_frame(pid, frame(capture=10), origin + 100_000_000, new_lease)[
            "frame_idx"
        ]
        == 0
    )
    assert source.subscription.frames.qsize() == 1
    with pytest.raises(ValueError, match="replaced"):
        hub.accept_frame(pid, frame(seq=1, capture=20), origin + 110_000_000, lease)
    with pytest.raises(ValueError, match="without gaps"):
        hub.accept_frame(pid, frame(seq=2, capture=20), origin + 110_000_000, new_lease)
    _, stamp = source.read(threading.Event())
    assert stamp == origin + 10_000_000
    assert source.frame_metadata["phone_host_receive_ns"] == origin + 90_000_000
    hub.progress(
        pid,
        new_lease,
        dict(type="heartbeat", capture_ms=30, last_sequence=0, buffered_frames=0),
    )
    assert source.drained_through(origin + 20_000_000)
    source.close()


def test_replay_backpressure_does_not_drop_frames(hub, monkeypatch):
    monkeypatch.setitem(phone.DEFAULTS, "camera_queue_size", 1)
    pairing = hub.pair(dict(name="Phone", width=320, height=240, fps=20))
    ready, origin = connect_resumable(hub, pairing)
    pid, lease = pairing["id"], ready["connection_id"]
    source = sources.CameraSource(
        CameraConfig(device=f"phone://{pid}", width=320, height=240, fps=20)
    )
    source.enable_recording()
    hub.accept_frame(pid, frame(), origin + 10_000_000, lease)
    accepted = threading.Event()

    def replay():
        hub.accept_frame(pid, frame(seq=1, capture=20), origin + 20_000_000, lease)
        accepted.set()

    worker = threading.Thread(target=replay)
    worker.start()
    try:
        assert not accepted.wait(0.2)
        source.read(threading.Event())
        assert accepted.wait(1)
        source.read(threading.Event())
        assert source.transport_stats["phone_queue_drops"] == 0
    finally:
        source.close()
        worker.join(2)


@pytest.mark.parametrize("recover", [True, False])
def test_session_stops_offline_then_recovers_buffered_capture_times(
    config, tmp_path, hub, monkeypatch, recover
):
    monkeypatch.setitem(
        phone.DEFAULTS, "phone_sync_timeout_seconds", 4 if recover else 0.4
    )
    monkeypatch.setitem(phone.DEFAULTS, "stall_timeout_seconds", 0.3)
    pairing = hub.pair(dict(name="Phone", width=320, height=240, fps=20))
    ready, origin = connect_resumable(hub, pairing)
    pid, lease = pairing["id"], ready["connection_id"]
    config.camera = CameraConfig(device=f"phone://{pid}", width=320, height=240, fps=20)
    config.duration_seconds = 0.8
    recorder = Recorder(config, tmp_path)
    errors, captured = [], []
    stop = threading.Event()

    def produce():
        nonlocal lease
        backlog, seq = [], 0
        disconnected = False
        try:
            while not stop.wait(0.05):
                capture = (time.monotonic_ns() - origin) / 1e6
                payload = frame(seq=seq, capture=capture)
                captured.append((seq, origin + int(capture * 1e6)))
                seq += 1
                if (
                    recorder.start_ns
                    and not disconnected
                    and time.monotonic_ns() > recorder.start_ns + 200_000_000
                ):
                    disconnected = True
                    hub.disconnect(pid, lease)
                if disconnected and (
                    not recover
                    or recorder.end_ns is None
                    or time.monotonic_ns() < recorder.end_ns + 300_000_000
                ):
                    backlog.append(payload)
                    continue
                if not hub.phones[pid].connected:
                    lease = hub.connect_resumable(
                        pid, {"resume_token": ready["resume_token"]}, [320, 240]
                    )["connection_id"]
                backlog.append(payload)
                for saved in backlog:
                    hub.accept_frame(pid, saved, time.monotonic_ns(), lease)
                backlog.clear()
                hub.progress(
                    pid,
                    lease,
                    dict(
                        type="heartbeat",
                        capture_ms=capture,
                        last_sequence=seq - 1,
                        buffered_frames=0,
                    ),
                )
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=produce)
    thread.start()
    try:
        result = recorder.run()
    finally:
        stop.set()
        thread.join(3)
    assert not errors
    if not recover:
        assert result["status"] == "incomplete"
        assert any(
            "Timed out waiting for buffered phone frames" in e for e in result["errors"]
        )
        assert result["statistics"]["camera"]["frames_recorded"] > 0
        assert (tmp_path / "sessions/test_session/raw/.video.mp4.tmp").is_file()
        assert all(r["written"] > 0 for r in result["statistics"]["receivers"])
        return
    assert result["status"] == "complete", result["errors"]
    assert result["statistics"]["camera"]["phone_reconnects"] == 1
    rows = pq.read_table(
        tmp_path / "sessions/test_session/raw/video_frames.parquet"
    ).to_pylist()
    expected = [
        seq
        for seq, stamp in captured
        if result["start_timestamp_ns"] <= stamp <= result["stop_timestamp_ns"]
    ]
    assert [r["phone_frame_idx"] for r in rows] == expected
    assert any(r["host_timestamp_ns"] > result["stop_timestamp_ns"] for r in rows)
    assert all(r["capture_timestamp_ns"] <= result["stop_timestamp_ns"] for r in rows)
    assert all(
        r["capture_timestamp_ns"] == r["phone_estimated_host_capture_ns"] for r in rows
    )
    assert rows[-1]["video_pts_s"] > 0.5
    assert result["statistics"]["camera"]["queue_drops"] == 0


def test_protocol2_socket_clock_and_resume_after_lost_ready(hub, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATA", tmp_path)
    monkeypatch.setattr(main, "recorder", None)
    pairing = hub.pair(dict(name="Phone", width=320, height=240, fps=20))
    pid = pairing["id"]
    with TestClient(main.app) as client:
        with client.websocket_connect("/phone/stream") as ws:
            ws.send_json(dict(id=pid, protocol=2, token=pairing["token"]))
            settings = ws.receive_json()
            # The phone persists its resume secret before consuming the pairing.
            assert settings["resume_token"]
        with client.websocket_connect("/phone/stream") as ws:
            ws.send_json(
                dict(id=pid, protocol=2, resume_token=settings["resume_token"])
            )
            assert ws.receive_json()["type"] == "settings"
            ws.send_json(dict(type="ready", native_resolution=[320, 240]))
            assert not ws.receive_json()["clock_ready"]
            ws.send_json(dict(type="clock", client_ms=0))
            probe = ws.receive_json()
            ws.send_json(
                dict(type="clock_commit", probe=probe["probe"], client_mid_ms=1)
            )
            assert ws.receive_json()["type"] == "synced"
            ws.send_bytes(frame())
            assert ws.receive_json()["frame_idx"] == 0
        with client.websocket_connect("/phone/stream") as ws:
            ws.send_json(
                dict(id=pid, protocol=2, resume_token=settings["resume_token"])
            )
            ws.receive_json()
            ws.send_json(dict(type="ready", native_resolution=[320, 240]))
            assert ws.receive_json()["last_sequence"] == 0
            ws.send_bytes(frame())
            assert ws.receive_json()["frame_idx"] == 0
            ws.send_json(
                dict(type="finish", capture_ms=20, last_sequence=0, buffered_frames=0)
            )
            assert ws.receive_json()["type"] == "finished"
