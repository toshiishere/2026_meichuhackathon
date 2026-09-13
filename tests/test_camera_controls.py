from types import SimpleNamespace
import pytest
from apps.common.schemas import CameraConfig
from apps.hardware_service.app import sources


@pytest.mark.parametrize("fixed", [True, False])
def test_camera_applies_exposure_frame_rate_policy_and_reports_negotiated_mode(
    monkeypatch, fixed
):
    monkeypatch.setattr(sources, "MODE", "real")
    monkeypatch.setattr(sources, "cameras", lambda: [{"device": "/dev/video0"}])
    cv = sources.cv2

    class Capture:
        def __init__(self, *args):
            self.values = {}
            self.closed = False

        def isOpened(self):
            return True

        def set(self, key, value):
            self.values[key] = value
            return True

        def get(self, key):
            return self.values.get(key, 0)

        def release(self):
            self.closed = True

    capture = Capture()
    monkeypatch.setattr(cv, "VideoCapture", lambda *args: capture)
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout="exposure_dynamic_framerate (bool): default=0 value=1",
        )

    monkeypatch.setattr(sources.subprocess, "run", run)
    source = sources.CameraSource(
        CameraConfig(
            device="/dev/video0", fps=30, width=1280, height=720, fixed_frame_rate=fixed
        )
    )
    try:
        assert (
            calls[-1][-1]
            == f"--set-ctrl=exposure_dynamic_framerate={0 if fixed else 1}"
        )
        assert source.diagnostics["dynamic_framerate_disabled"] is fixed
        assert source.diagnostics["negotiated_fps"] == 30
        assert source.diagnostics["pixel_format"] == "MJPG"
        assert source.diagnostics["negotiated_resolution"] == [1280, 720]
        assert "value=1" in source.diagnostics["controls_before"]
    finally:
        source.close()
    assert capture.closed
