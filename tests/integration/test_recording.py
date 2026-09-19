import csv
import hashlib
import io
import json
import threading
import time
import av
import pyarrow.parquet as pq
import pytest
import zstandard
from apps.hardware_service.app.recorder import Recorder
from apps.common.storage import read_session, rebuild_manifest


def read_csv(path):
    with path.open("rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as stream:
        return list(csv.DictReader(io.TextIOWrapper(stream)))


@pytest.mark.parametrize("external_sender", [False, True])
def test_multi_receiver_video_timestamps_and_immutable_raw(
    config, tmp_path, external_sender
):
    if external_sender:
        config.sender = None
    recorder = Recorder(config, tmp_path)
    metadata = recorder.run()
    assert metadata["status"] == "complete", metadata["errors"]
    assert metadata["hardware_mode"] == "synthetic"
    assert (metadata["configuration"]["sender"] is None) == external_sender
    root = tmp_path / "sessions" / config.session_id
    frame_rows = pq.read_table(root / "raw/video_frames.parquet").to_pylist()
    assert len(frame_rows) >= 15
    stamps = [r["host_timestamp_ns"] for r in frame_rows]
    assert stamps == sorted(set(stamps))
    assert stamps[0] >= metadata["start_timestamp_ns"]
    pts = []
    with av.open(str(root / "raw/video.mp4")) as video:
        for frame in video.decode(video=0):
            pts.append(float(frame.pts * frame.time_base))
    assert len(pts) == len(frame_rows)
    assert pts == pytest.approx([r["video_pts_s"] for r in frame_rows], abs=2e-6)
    # Recordings carry a segment index, so a browser can report a duration and
    # seek without first reading the whole file.
    from apps.training_service.app.playback import needs_index

    assert not needs_index(root / "raw/video.mp4")
    raw_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / "raw").iterdir()
    }
    receivers = []
    for name in ["rx_left", "rx_right"]:
        rows = read_csv(root / f"raw/csi_{name}.csv.zst")
        assert 70 < len(rows) < 130
        assert all(
            int(r["host_timestamp_ns"]) >= metadata["start_timestamp_ns"] for r in rows
        )
        assert all(
            int(r["host_timestamp_ns"]) <= metadata["stop_timestamp_ns"] for r in rows
        )
        assert all(
            r["tx_seq"] == r["id"] and int(r["len"]) == len(json.loads(r["data"]))
            for r in rows
        )
        assert rows[0]["raw_line"].startswith("CSI_DATA,")
        assert "gain_agc" in rows[0]
        receivers.append({int(r["tx_seq"]): int(r["host_timestamp_ns"]) for r in rows})
    common = receivers[0].keys() & receivers[1].keys()
    assert len(common) > 65
    assert (
        max(abs(receivers[0][seq] - receivers[1][seq]) for seq in common) < 100_000_000
    )
    assert any(r["sequence_gaps"] > 0 for r in metadata["statistics"]["receivers"])
    assert not (root / "labels").exists() and not (root / "processed").exists()
    assert len(pq.read_table(tmp_path / "manifest.parquet")) == 1
    with pytest.raises(FileExistsError):
        Recorder(config, tmp_path).run()
    rebuild_manifest(tmp_path)
    assert raw_hashes == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / "raw").iterdir()
    }


def test_manual_stop_without_duration(config, tmp_path):
    config.duration_seconds = None
    stop = threading.Event()
    recorder = Recorder(config, tmp_path, stop)
    thread = threading.Thread(target=recorder.run)
    thread.start()
    deadline = time.monotonic() + 10
    while recorder.state != "recording" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert recorder.state == "recording"
    time.sleep(0.5)
    stop.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert read_session(tmp_path, config.session_id)["status"] == "complete"
    assert not any(t.is_alive() for t in recorder.threads)


def test_encoder_failure_preserves_incomplete(config, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("simulated encoder disk failure")

    monkeypatch.setattr("apps.hardware_service.app.recorder.av.open", fail)
    metadata = Recorder(config, tmp_path).run()
    assert metadata["status"] == "incomplete"
    assert any("simulated encoder disk failure" in e for e in metadata["errors"])
    assert (tmp_path / "sessions/test_session/metadata.json").exists()
    assert not (tmp_path / "sessions/test_session/raw/video.mp4").exists()


def test_camera_disconnect_stops_all_streams(config, tmp_path, monkeypatch):
    from apps.hardware_service.app.sources import CameraSource

    original = CameraSource.read
    calls = 0

    def read(self, stop):
        nonlocal calls
        calls += 1
        if calls > 7:
            raise OSError("camera disconnected")
        return original(self, stop)

    monkeypatch.setattr(CameraSource, "read", read)
    recorder = Recorder(config, tmp_path)
    result = recorder.run()
    assert result["status"] == "incomplete"
    assert any("camera disconnected" in e for e in result["errors"])
    assert not any(t.is_alive() for t in recorder.threads)
    assert list((tmp_path / "sessions/test_session/raw").glob(".*.tmp"))


def test_overflow_is_visible_and_degraded(config, tmp_path, monkeypatch):
    from apps.hardware_service.app.recorder import DEFAULTS

    monkeypatch.setitem(DEFAULTS, "serial_queue_size", 1)
    original = Recorder._serial_writer

    def slow(self, *args):
        time.sleep(0.3)
        return original(self, *args)

    monkeypatch.setattr(Recorder, "_serial_writer", slow)
    result = Recorder(config, tmp_path).run()
    assert result["status"] == "complete", result["errors"]
    assert result["quality"] == "degraded"
    assert all(r["queue_drops"] > 0 for r in result["statistics"]["receivers"])


def test_startup_crash_recovery_keeps_artifacts(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from apps.hardware_service.app import main

    root = tmp_path / "sessions/interrupted"
    (root / "raw").mkdir(parents=True)
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "session_id": "interrupted",
                "status": "recording",
            }
        )
    )
    (root / "raw/.video.mp4.tmp").write_bytes(b"partial video")
    monkeypatch.setattr(main, "DATA", tmp_path)
    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 200
        assert read_session(tmp_path, "interrupted")["status"] == "incomplete"
        assert (root / "raw/.video.mp4.tmp").read_bytes() == b"partial video"


def test_recording_requires_fresh_serial_readiness(config, tmp_path, monkeypatch):
    from apps.hardware_service.app.sources import SerialSource
    from apps.hardware_service.app.recorder import DEFAULTS

    monkeypatch.setitem(DEFAULTS, "stall_timeout_seconds", 0.05)

    def silent(self, stop):
        stop.wait(0.01)
        return None

    monkeypatch.setattr(SerialSource, "read", silent)
    recorder = Recorder(config, tmp_path)
    result = recorder.run()
    assert result["status"] == "incomplete"
    assert recorder.start_ns is None
    assert any("no valid CSI after opening" in error for error in result["errors"])
