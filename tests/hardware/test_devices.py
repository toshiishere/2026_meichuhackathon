"""Explicit read-only physical checks. Flashing is intentionally a UI operation."""

import os
import pytest

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        os.getenv("RUN_HARDWARE_TESTS") != "1", reason="Opt-in hardware tests"
    ),
]


def test_serial_discovery():
    from serial.tools import list_ports

    assert list(list_ports.comports()), "No USB serial devices connected"


def test_camera_capture():
    import cv2

    path = os.environ.get("TEST_CAMERA", "/dev/video0")
    camera = cv2.VideoCapture(path, cv2.CAP_V4L2)
    try:
        assert camera.isOpened()
        ok, frame = camera.read()
        assert ok and frame.size > 0
    finally:
        camera.release()
