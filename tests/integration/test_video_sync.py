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


def fragmented_video(path, seconds=3, fps=10, sidx=False):
    """A recording in the collector's own container format."""
    import av

    flags = "frag_keyframe+empty_moov+default_base_moof" + (
        "+global_sidx" if sidx else ""
    )
    with av.open(str(path), "w", options={"movflags": flags}) as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "35", "g": str(fps)}
        for i in range(seconds * fps):
            frame = np.full((48, 64, 3), i % 255, dtype=np.uint8)
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(frame, format="bgr24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_unindexed_recording_is_indexed_into_derived_leaving_raw_untouched(tmp_path):
    import hashlib
    import json
    from apps.training_service.app import playback

    session = tmp_path / "session"
    (session / "raw").mkdir(parents=True)
    source = session / "raw/video.mp4"
    fragmented_video(source)
    assert playback.needs_index(source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    assert playback.ensure_playable_video(session) == "derived/video.mp4"
    indexed = session / "derived/video.mp4"
    assert indexed.is_file() and not playback.needs_index(indexed)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before

    import av

    with av.open(str(indexed)) as container:
        # A front index: duration and frame count without scanning fragments.
        assert container.streams.video[0].frames > 0
        assert float(container.duration) > 0
    record = json.loads((session / "derived/video.json").read_text())
    assert record["identity"]["bytes"] == source.stat().st_size

    # The copy is reused, and rebuilt when the recording itself changes.
    built = indexed.stat().st_mtime_ns
    assert playback.ensure_playable_video(session) == "derived/video.mp4"
    assert indexed.stat().st_mtime_ns == built
    fragmented_video(source, seconds=2)
    assert playback.ensure_playable_video(session) == "derived/video.mp4"
    assert indexed.stat().st_mtime_ns != built


def test_indexed_recordings_play_from_raw_without_a_copy(tmp_path):
    from apps.training_service.app import playback

    session = tmp_path / "session"
    (session / "raw").mkdir(parents=True)
    # What the collector writes now: fragmented, but with a segment index.
    fragmented_video(session / "raw/video.mp4", sidx=True)
    assert not playback.needs_index(session / "raw/video.mp4")
    assert playback.ensure_playable_video(session) == "raw/video.mp4"
    assert not (session / "derived").exists()
