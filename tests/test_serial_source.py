import os
import pty
import threading
from apps.hardware_service.app import sources


def test_serial_partial_lines_survive_timeouts(monkeypatch):
    master, slave = pty.openpty()
    port = os.ttyname(slave)
    monkeypatch.setattr(sources, "MODE", "real")
    monkeypatch.setattr(sources, "validate_port", lambda _: {"identity": "test"})
    source = sources.SerialSource(port, 921600)
    try:
        os.write(master, b"CSI_DATA,partial")
        assert source.read(threading.Event()) is None
        assert source.buffer == b"CSI_DATA,partial"
        os.write(master, b",complete\n")
        line, stamp = source.read(threading.Event())
        assert line == b"CSI_DATA,partial,complete"
        assert stamp > 0
        assert source.buffer == b""
    finally:
        source.close()
        os.close(master)
        os.close(slave)


def test_usb_identity_without_location_stays_stable(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(sources, "MODE", "real")
    board = SimpleNamespace(
        device="/dev/ttyUSB0",
        vid=123,
        pid=456,
        serial_number="unique",
        location=None,
        manufacturer="vendor",
        product="board",
        description="board",
    )
    monkeypatch.setattr(sources.list_ports, "comports", lambda: [board])
    first = sources.serial_devices()[0]["identity"]
    board.device = "/dev/ttyUSB9"
    assert sources.serial_devices()[0]["identity"] == first == "usb:123:456:unique"


def test_discovery_only_lists_supported_usb_acm_ports(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(sources, "MODE", "real")

    def port(name):
        return SimpleNamespace(
            device=name,
            vid=1,
            pid=2,
            serial_number=name,
            location=None,
            manufacturer="",
            product="",
            description="",
        )

    names = [
        "/dev/ttyS0",
        "/dev/ttyS31",
        "/dev/ttyUSB0",
        "/dev/ttyUSB9",
        "/dev/ttyACM0",
        "/dev/ttyACM9",
        "/dev/ttyUSB10",
        "/dev/ttyACM10",
    ]
    monkeypatch.setattr(
        sources.list_ports, "comports", lambda: [port(n) for n in names]
    )
    assert [p["device"] for p in sources.serial_devices()] == names[2:6]
    from apps.common.schemas import SerialRequest
    from pydantic import ValidationError
    import pytest

    for name in [names[0], names[-1]]:
        with pytest.raises(ValidationError):
            SerialRequest(port=name)
