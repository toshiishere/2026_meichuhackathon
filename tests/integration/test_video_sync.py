import time

import numpy as np
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


def test_camera_runs_at_its_own_rate_and_keeps_history_for_the_fused_window(
    tmp_path,
):
    """Alignment can only be as fine as the preview's frame interval."""
    from apps.hardware_service.app.jobs import Jobs
    from apps.common.schemas import DeployCaptureRequest

    handle = DeployCapture(Jobs(tmp_path))
    token = handle.start(
        DeployCaptureRequest(
            camera=dict(device="synthetic://camera", fps=20, width=320, height=240)
        )
    )["capture_id"]
    try:
        assert handle.frames.maxlen == 20 * 6  # Six seconds of frames.
        deadline = time.monotonic() + 5
        while len(handle.frames) < 20 and time.monotonic() < deadline:
            time.sleep(0.02)
        stamps = [stamp for _, stamp in handle.frames]
        assert len(stamps) >= 20
        gaps = np.diff(stamps)
        # ~50 ms apart at 20 FPS, not the 80 ms a fixed preview throttle gave.
        assert np.median(gaps) < 70_000_000
        # A window end already a second in the past still resolves to its frame.
        past = stamps[-1] - 400_000_000
        jpeg, served = handle.frame(token, past)
        assert abs(served - past) <= 60_000_000
        assert served in stamps and jpeg
    finally:
        handle.close()
