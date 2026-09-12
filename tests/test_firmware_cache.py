import threading
import pytest
from apps.hardware_service.app import firmware
from apps.common.schemas import FlashRequest


def test_build_cache_and_failed_rebuild_invalidation(tmp_path, monkeypatch):
    monkeypatch.setattr(firmware, "DATA", tmp_path)
    monkeypatch.setattr(firmware, "MODE", "real")
    commands = []

    def run(args, cwd, log, stop, **kwargs):
        commands.append(args)
        return "build succeeded"

    monkeypatch.setattr(firmware, "run_command", run)
    request = FlashRequest(
        port="/dev/ttyUSB0", target="esp32", firmware="csi-send", operation="build"
    )
    first = firmware.flash(request, lambda _: None, threading.Event())
    assert commands == [["idf.py", "build"]]
    assert first["flashed"] is False and first["identity"] is None
    firmware.flash(request, lambda _: None, threading.Event())
    assert len(commands) == 1

    def fail(*args, **kwargs):
        raise RuntimeError("build failed")

    monkeypatch.setattr(firmware, "run_command", fail)
    with pytest.raises(RuntimeError):
        firmware.flash(
            request.model_copy(update={"operation": "rebuild"}),
            lambda _: None,
            threading.Event(),
        )
    assert not (
        tmp_path / "app/firmware_cache" / first["cache_key"] / "build_complete.json"
    ).exists()
