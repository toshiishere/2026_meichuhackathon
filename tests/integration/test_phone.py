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
        r["schema_version"] == "1.1" and r["phone_frame_idx"] is not None for r in rows
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
