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


def test_generated_build_trees_do_not_change_firmware_cache(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    source = root / "firmware/esp-csi/csi_recv"
    (source / "main").mkdir(parents=True)
    (source / "main/app_main.c").write_text("source-v1")
    (source / "sdkconfig").write_text('CONFIG_IDF_TARGET="esp32c3"')
    (root / "firmware/provenance.json").write_text('{"esp_csi_git_commit":"baseline"}')
    monkeypatch.setattr(firmware, "ROOT", root)
    monkeypatch.setattr(firmware, "DATA", tmp_path / "data")
    monkeypatch.setattr(firmware, "MODE", "real")
    monkeypatch.setattr(firmware, "run_command", lambda *args, **kwargs: "ok")
    request = FlashRequest(
        port="/dev/ttyACM0", target="esp32c3", firmware="csi-recv", operation="build"
    )
    first = firmware.flash(request, lambda _: None, threading.Event())
    (source / "build").mkdir()
    (source / "build/generated.c").symlink_to("/nonexistent-generated-source")
    (source / "managed_components").mkdir()
    (source / "managed_components/generated").write_text("downloaded build state")
    second = firmware.flash(request, lambda _: None, threading.Event())
    assert first["cache_key"] == second["cache_key"]
    (source / "main/app_main.c").write_text("source-v2")
    changed = firmware.flash(request, lambda _: None, threading.Event())
    assert first["cache_key"] != changed["cache_key"]
    project = tmp_path / "data/app/firmware_cache" / changed["cache_key"] / "project"
    assert (
        not (project / "build").exists()
        and not (project / "managed_components").exists()
    )


@pytest.mark.parametrize(
    "port,target,expected",
    [
        ("/dev/ttyACM0", "esp32c3", "usb"),
        (
            "/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit-id",
            "esp32c3",
            "usb",
        ),
        ("/dev/ttyUSB0", "esp32c3", "uart"),
        ("/dev/serial/by-id/usb-Silicon_Labs_CP2102-id", "esp32s3", "uart"),
        ("/dev/ttyUSB0", "esp32", "uart"),
    ],
)
def test_receiver_output_matches_physical_connection(port, target, expected):
    request = FlashRequest(
        port=port, target=target, firmware="csi-recv", operation="build"
    )
    assert firmware.receiver_transport(request) == expected


def test_output_choice_is_validated_and_saved_in_build_configuration(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(firmware, "DATA", tmp_path)
    monkeypatch.setattr(firmware, "MODE", "real")

    def run(args, cwd, *rest, **kwargs):
        defaults = (cwd / "sdkconfig.defaults").read_text()
        assert "CONFIG_CSI_OUTPUT_USB=y" in defaults
        assert "# CONFIG_CSI_OUTPUT_UART is not set" in defaults
        assert "CONFIG_CSI_OUTPUT_USB=y" in (cwd / "sdkconfig").read_text()

    monkeypatch.setattr(firmware, "run_command", run)
    request = FlashRequest(
        port="/dev/ttyACM0", target="esp32c3", firmware="csi-recv", operation="build"
    )
    result = firmware.flash(request, lambda _: None, threading.Event())
    assert result["csi_transport"] == "usb"
    with pytest.raises(ValueError, match="no native USB"):
        firmware.receiver_transport(
            request.model_copy(update={"target": "esp32", "csi_transport": "usb"})
        )
