import time
import pytest
from fastapi import HTTPException
from apps.hardware_service.app.deploy import DeployCapture
from apps.training_service.app.timeline import Timeline
from test_training import make_session


def test_replay_video_uses_capture_clock_not_phone_arrival_time(tmp_path):
    session = make_session(tmp_path)
    timeline = Timeline(session / "raw/video_frames.parquet")
    assert timeline.video_seconds(14_250_000_000) == 4.25
    assert timeline.video_seconds(1) == 0
    assert timeline.video_seconds(99_000_000_000) == 8
    assert timeline.host_ns(timeline.video_seconds(14_250_000_000)) == 14_250_000_000


def test_live_camera_selects_nearest_capture_timestamp_and_rejects_stale():
    capture = DeployCapture(None)
    capture.token = "token"
    now = time.monotonic_ns()
    capture.frames.extend([(b"earlier", now - 300_000_000), (b"latest", now)])
    capture.jpeg, capture.camera_stamp = capture.frames[-1]
    assert capture.frame("token", now - 280_000_000)[0] == b"earlier"
    assert capture.frame("token")[0] == b"latest"
    with pytest.raises(HTTPException, match="aligned"):
        capture.frame("token", now - 2_000_000_000)
    capture.stop.set()
    with pytest.raises(HTTPException):
        capture.frame("token")
