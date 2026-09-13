import hashlib
import json
import os
import shutil
import subprocess
import time
from apps.common.config import ROOT, DATA, MODE
from apps.common.storage import atomic_json
from .sources import validate_port

IDF_VERSION = "5.5.0"
GENERATED_DIRS = {"build", "managed_components", ".git", "__pycache__"}


def run_command(args, cwd, log, stop, timeout=1200):
    log("Command: " + " ".join(args))
    # Log directly to disk: noisy builds cannot deadlock on unread pipe buffers.
    import tempfile

    with tempfile.TemporaryFile(mode="w+") as output:
        process = subprocess.Popen(
            args,
            cwd=cwd,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        started, offset = time.monotonic(), 0
        try:
            while process.poll() is None:
                chunk_bytes = os.pread(output.fileno(), 1024 * 1024, offset)
                offset += len(chunk_bytes)
                chunk = chunk_bytes.decode(errors="replace")
                if chunk:
                    log(chunk.rstrip())
                if stop.is_set() or time.monotonic() - started > timeout:
                    import signal

                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise RuntimeError("Command cancelled or timed out")
                time.sleep(0.2)
            output.seek(offset)
            chunk = output.read()
            if chunk:
                log(chunk.rstrip())
            if process.returncode:
                raise RuntimeError(
                    f"{args[0]} exited with status {process.returncode}; inspect job logs"
                )
            output.seek(0)
            return output.read()[-30000:]
        finally:
            if process.poll() is None:
                import signal

                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def probe(port, log, stop):
    device = validate_port(port)
    if MODE == "synthetic":
        log("Synthetic probe; no physical board was accessed")
        return dict(
            device=device,
            simulated=True,
            chip="Simulated ESP32",
            output="Synthetic probe only",
        )
    output = run_command(
        ["esptool.py", "--port", port, "flash_id"], ROOT, log, stop, timeout=30
    )
    return dict(
        device=device,
        simulated=False,
        output=output,
        note="esptool may reset the board into the bootloader; probe does not write flash",
    )


def receiver_transport(request):
    if request.csi_transport != "auto":
        transport = request.csi_transport
    else:
        # Stable by-id paths preserve the connection type across tty renumbering.
        transport = (
            "usb"
            if (
                request.target != "esp32"
                and (
                    request.port.startswith("/dev/ttyACM") or "USB_JTAG" in request.port
                )
            )
            else "uart"
        )
    if transport == "usb" and request.target == "esp32":
        raise ValueError("ESP32 has no native USB Serial/JTAG; select UART output")
    return transport


def set_sdk_option(path, key, enabled):
    lines = path.read_text().splitlines() if path.exists() else []
    lines = [
        line
        for line in lines
        if not line.startswith(key + "=") and line != f"# {key} is not set"
    ]
    lines.append(f"{key}=y" if enabled else f"# {key} is not set")
    path.write_text("\n".join(lines) + "\n")


def flash(request, log, stop):
    device = validate_port(request.port) if request.operation == "flash" else None
    if MODE == "synthetic":
        raise ValueError(
            "Firmware building/flashing is unavailable in synthetic mode; switch to the real hardware Compose configuration"
        )
    if request.firmware == "blink" and (
        request.gpio is None or request.led_type == "none"
    ):
        raise ValueError(
            "Blink unavailable without a usable LED; select LED type and GPIO from the board pinout"
        )
    source = (
        ROOT
        / "firmware"
        / (
            "blink_identify"
            if request.firmware == "blink"
            else f"esp-csi/{request.firmware.replace('-', '_')}"
        )
    )
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*")):
        if path.is_file() and not GENERATED_DIRS.intersection(
            path.relative_to(source).parts
        ):
            digest.update(str(path.relative_to(source)).encode())
            digest.update(path.read_bytes())
    provenance = json.loads((ROOT / "firmware/provenance.json").read_text())
    build_config = dict(
        firmware=request.firmware,
        target=request.target,
        gpio=request.gpio,
        led_type=request.led_type,
        active_low=request.active_low,
        idf_version=IDF_VERSION,
        esp_csi_git_commit=provenance["esp_csi_git_commit"],
        source_sha256=digest.hexdigest(),
    )
    if request.firmware == "csi-recv":
        build_config["csi_transport"] = receiver_transport(request)
    key = hashlib.sha256(json.dumps(build_config, sort_keys=True).encode()).hexdigest()[
        :24
    ]
    cache = DATA / "app/firmware_cache" / key
    if request.operation == "clean-build" and cache.exists():
        shutil.rmtree(cache)
    marker = cache / "build_complete.json"
    if not marker.exists() or request.operation in {"rebuild", "clean-build"}:
        marker.unlink(missing_ok=True)
        project = cache / "project"
        if project.exists():
            shutil.rmtree(project)
        project.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, project, ignore=shutil.ignore_patterns(*GENERATED_DIRS))
        # Keep full known-working SDK config for its original target. For another
        # target, derive from the same sdkconfig.defaults (target-specific values
        # cannot safely be copied across chips).
        sdk = project / "sdkconfig"
        preserved = (
            sdk.exists() and f'CONFIG_IDF_TARGET="{request.target}"' in sdk.read_text()
        )
        if not preserved:
            sdk.unlink(missing_ok=True)
        if request.firmware == "csi-recv":
            output = build_config["csi_transport"]
            for config_file in [
                project / "sdkconfig.defaults",
                *([sdk] if preserved else []),
            ]:
                set_sdk_option(config_file, "CONFIG_CSI_OUTPUT_USB", output == "usb")
                set_sdk_option(config_file, "CONFIG_CSI_OUTPUT_UART", output == "uart")
        if request.firmware == "blink":
            sdk.unlink(missing_ok=True)
            defaults = project / "sdkconfig.defaults"
            defaults.write_text(
                defaults.read_text()
                + f"\nCONFIG_BLINK_GPIO={request.gpio}\nCONFIG_BLINK_LED_{request.led_type.upper()}=y\nCONFIG_BLINK_ACTIVE_LOW={'y' if request.active_low else 'n'}\n"
            )
        if not preserved or request.firmware == "blink":
            run_command(["idf.py", "set-target", request.target], project, log, stop)
        run_command(["idf.py", "build"], project, log, stop)
        atomic_json(marker, build_config)
        log(f"Build cached as {key}")
    else:
        log(f"Using cached firmware build {key}")
    if request.operation == "flash":
        run_command(
            ["idf.py", "-p", request.port, "flash"], cache / "project", log, stop
        )
    return dict(
        **build_config,
        cache_key=key,
        operation=request.operation,
        flashed=request.operation == "flash",
        simulated=False,
        port=request.port,
        identity=device["identity"] if device else None,
    )
