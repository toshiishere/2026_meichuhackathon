import concurrent.futures
import shutil
import threading
import time
import tempfile
from pathlib import Path
from apps.common.config import DEFAULTS
from .csi import parse_csi, SequenceTracker
from .sources import SerialSource, CameraSource, validate_port


def serial_test(request, stop=None):
    stop = stop or threading.Event()
    source = SerialSource(request.port, request.baud_rate, request.expected_rate_hz)
    start = time.monotonic()
    tracker, count, errors, noise, rssi = SequenceTracker(), 0, 0, 0, 0
    monitor = []
    try:
        while not stop.is_set() and time.monotonic() - start < request.seconds:
            value = source.read(stop)
            if value is None:
                continue
            raw, timestamp = value
            line = raw.decode("utf-8", errors="replace")
            if len(monitor) < 100:
                monitor.append({"host_timestamp_ns": timestamp, "line": line[:4096]})
            try:
                packet = parse_csi(line)
                if packet:
                    tracker.update(packet["tx_seq"])
                    count += 1
                    rssi += packet["rssi"]
                else:
                    noise += 1
            except ValueError:
                errors += 1
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
        passed=rate >= request.expected_rate_hz * request.min_rate_ratio
        and errors == 0,
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
    devices = [validate_port(x.port) for x in [config.sender, *config.receivers]]
    if len({x["identity"] for x in devices}) != len(devices):
        raise ValueError("Selected ports resolve to the same physical device")
    for selected, current in zip([config.sender, *config.receivers], devices):
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
        sender=devices[0],
        receivers=receivers,
        camera=cam,
        storage=storage,
    )
