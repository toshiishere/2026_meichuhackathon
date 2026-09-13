import concurrent.futures
import shutil
import threading
import time
import tempfile
from pathlib import Path
from apps.common.config import DEFAULTS
from .csi import parse_csi, SequenceTracker, TransportTracker
from .sources import SerialSource, CameraSource, validate_port


def serial_test(request, stop=None):
    stop = stop or threading.Event()
    source = SerialSource(request.port, request.baud_rate, request.expected_rate_hz)
    start = time.monotonic()
    tracker, count, errors, noise, rssi = SequenceTracker(), 0, 0, 0, 0
    monitor = []
    transport = {}
    transport_tracker = TransportTracker()
    try:
        while not stop.is_set() and time.monotonic() - start < request.seconds:
            value = source.read(stop)
            if value is None:
                continue
            raw, timestamp = value
            line = raw.decode("utf-8", errors="replace")
            try:
                packet = parse_csi(raw)
                if packet:
                    tracker.update(packet["tx_seq"])
                    count += 1
                    transport.update(transport_tracker.update(packet))
                    if packet["firmware_layout"] == "binary_v1":
                        line = f"CSI_BINARY seq={packet['tx_seq']} samples={packet['len']} rssi={packet['rssi']} firmware_queue_drops_total={packet['firmware_queue_drops_total']}"
                    rssi += packet["rssi"]
                    for key in (
                        "firmware_layout",
                        "firmware_received_total",
                        "firmware_queue_drops_total",
                        "firmware_invalid_total",
                    ):
                        if key in packet:
                            transport[key] = packet[key]
                else:
                    noise += 1
            except ValueError:
                errors += 1
            if len(monitor) < 100:
                monitor.append({"host_timestamp_ns": timestamp, "line": line[:4096]})
    finally:
        source.close()
    elapsed = time.monotonic() - start
    rate = count / elapsed
    return dict(
        port=request.port,
        packets=count,
        duration_seconds=elapsed,
        rate_hz=rate,
        parse_errors=errors,
        non_csi_lines=noise,
        average_rssi=rssi / count if count else None,
        **tracker.snapshot(),
        transport=transport,
        passed=rate >= request.expected_rate_hz * request.min_rate_ratio
        and errors == 0
        and not transport.get("firmware_queue_drops", 0)
        and not transport.get("firmware_invalid", 0),
        serial_monitor=monitor,
    )


def camera_test(config, seconds=3, stop=None):
    stop = stop or threading.Event()
    source = CameraSource(config)
    start = time.monotonic()
    first = last = None
    count, size = 0, None
    try:
        while not stop.is_set() and time.monotonic() - start < seconds:
            result = source.read(stop)
            if result:
                frame, last = result
                first = last if first is None else first
                count += 1
                size = [frame.shape[1], frame.shape[0]]
    finally:
        source.close()
    fps = (count - 1) * 1e9 / (last - first) if count > 1 and last > first else 0
    return dict(
        device=config.device,
        frames=count,
        measured_fps=fps,
        requested_fps=config.fps,
        duration_seconds=time.monotonic() - start,
        diagnostics=getattr(source, "diagnostics", {}),
        rate_note=(
            "Measured frame rate is below the requested rate; inspect negotiated mode, exposure controls, lighting, and transport drops."
            if fps < config.fps * DEFAULTS["min_rate_ratio"]
            else "Measured rate meets the requested threshold."
        ),
        resolution=size,
        last_frame_timestamp_ns=last,
        passed=count > 1
        and size == [config.width, config.height]
        and fps >= config.fps * DEFAULTS["min_rate_ratio"],
    )


def preflight(config, root: Path, stop=None):
    from apps.common.schemas import SerialRequest

    stop = stop or threading.Event()
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    with tempfile.TemporaryFile(dir=root) as f:
        f.write(b"collection preflight")
        f.flush()
    storage = dict(
        free_bytes=free,
        minimum_free_bytes=DEFAULTS["minimum_free_bytes"],
        passed=free >= DEFAULTS["minimum_free_bytes"],
    )
    selected_devices = ([config.sender] if config.sender else []) + config.receivers
    devices = [validate_port(x.port) for x in selected_devices]
    if len({x["identity"] for x in devices}) != len(devices):
        raise ValueError("Selected ports resolve to the same physical device")
    for selected, current in zip(selected_devices, devices):
        if selected.identity and selected.identity != current["identity"]:
            raise ValueError(
                f"{selected.logical_name} identity changed; refresh device assignments"
            )

    def check(fn, *args):
        try:
            return fn(*args)
        except Exception as e:
            return {"passed": False, "error": str(e)}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(config.receivers) + 1
    ) as pool:
        rx = [
            pool.submit(
                check,
                serial_test,
                SerialRequest(
                    port=r.port,
                    baud_rate=config.baud_rate,
                    expected_rate_hz=config.expected_rate_hz,
                    min_rate_ratio=config.min_rate_ratio,
                    seconds=DEFAULTS["preflight_seconds"],
                ),
                stop,
            )
            for r in config.receivers
        ]
        camera = pool.submit(
            check, camera_test, config.camera, DEFAULTS["preflight_seconds"], stop
        )
        receivers = [
            {**f.result(), "logical_name": r.logical_name}
            for r, f in zip(config.receivers, rx)
        ]
        for receiver in receivers:
            receiver.pop("serial_monitor", None)
        cam = camera.result()
    return dict(
        passed=not stop.is_set()
        and storage["passed"]
        and cam["passed"]
        and all(r["passed"] for r in receivers),
        sender=devices[0] if config.sender else None,
        sender_connection="usb" if config.sender else "external",
        receivers=receivers,
        camera=cam,
        storage=storage,
    )
