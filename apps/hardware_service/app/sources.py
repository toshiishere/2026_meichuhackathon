import glob
import math
import random
import re
import time
from pathlib import Path
import cv2
import numpy as np
import serial
from serial.tools import list_ports
from apps.common.config import MODE, DEFAULTS
from .csi import LEGACY
from .phone import hub, PhoneCameraSource


def serial_devices():
    if MODE == "synthetic":
        return [
            dict(
                port=f"synthetic://{x}",
                device=f"synthetic://{x}",
                stable_path=f"synthetic://{x}",
                identity=f"synthetic-{x}",
                serial_number=f"SIM-{x}",
                manufacturer="Synthetic",
                product="Simulated ESP32",
                description="Synthetic device",
                vid=None,
                pid=None,
                location=None,
            )
            for x in ["tx", "rx0", "rx1"]
        ]
    stable = {}
    for p in sorted(Path("/dev/serial/by-id").glob("*")):
        stable.setdefault(str(p.resolve()), str(p))
    ports = [
        p
        for p in list_ports.comports()
        if re.fullmatch(r"/dev/tty(?:USB|ACM)[0-9]", p.device)
    ]
    serial_counts = {}
    for p in ports:
        if p.serial_number:
            serial_counts[p.serial_number] = serial_counts.get(p.serial_number, 0) + 1
    result = []
    for p in ports:
        path = stable.get(str(Path(p.device).resolve()))
        if p.serial_number and serial_counts[p.serial_number] == 1:
            identity = f"usb:{p.vid}:{p.pid}:{p.serial_number}"
        else:
            identity = path or (
                f"location:{p.location}:{p.vid}:{p.pid}" if p.location else p.device
            )
        result.append(
            dict(
                port=path or p.device,
                device=p.device,
                stable_path=path,
                identity=identity,
                serial_number=p.serial_number,
                manufacturer=p.manufacturer,
                product=p.product,
                description=p.description,
                vid=p.vid,
                pid=p.pid,
                location=p.location,
            )
        )
    return result


def validate_port(port):
    match = next(
        (
            d
            for d in serial_devices()
            if port in {d["port"], d["device"], d["stable_path"]}
        ),
        None,
    )
    if not match:
        raise ValueError(
            f"Serial device {port} is not currently discovered; reconnect it and refresh hardware"
        )
    return match


def cameras():
    if MODE == "synthetic":
        return [
            dict(
                device="synthetic://camera",
                name="Synthetic timing camera",
                formats="Generated frames; requested size and FPS",
            )
        ] + hub.discover()
    result = []
    import subprocess

    for device in sorted(glob.glob("/dev/video*")):
        name_file = Path("/sys/class/video4linux") / Path(device).name / "name"
        try:
            formats = subprocess.run(
                ["v4l2-ctl", "--device", device, "--list-formats-ext"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            details = formats.stdout or formats.stderr
        except (OSError, subprocess.TimeoutExpired) as e:
            details = str(e)
        result.append(
            dict(
                device=device,
                name=name_file.read_text().strip() if name_file.exists() else device,
                formats=details,
            )
        )
    return result + hub.discover()


class SerialSource:
    def __init__(self, port, baud_rate, rate=100, loss=0, seed=42):
        validate_port(port)
        self.port = port
        self.synthetic = MODE == "synthetic"
        self.rate, self.loss = rate, loss
        self.random = random.Random(seed)
        self.handle = None
        self.buffer = bytearray()
        self.discarding = False
        if not self.synthetic:
            # Avoid deliberate DTR/RTS resets during collection. Some USB drivers
            # may still pulse lines on open: preflight checks actual arrival.
            self.handle = serial.Serial(
                port=None,
                baudrate=baud_rate,
                timeout=DEFAULTS["serial_timeout_seconds"],
                exclusive=True,
            )
            self.handle.dtr = False
            self.handle.rts = False
            self.handle.port = port
            try:
                self.handle.open()
            except Exception:
                self.handle.close()
                raise
        self.arm(time.monotonic_ns())

    def arm(self, start_ns):
        self.start_ns = start_ns
        self.next_seq = 0
        self.buffer.clear()
        self.discarding = False
        if self.handle:
            self.handle.reset_input_buffer()

    def read(self, stop):
        if self.synthetic:
            seq = self.next_seq
            wait = (
                self.start_ns + int(seq * 1e9 / self.rate) - time.monotonic_ns()
            ) / 1e9
            if wait > 0 and stop.wait(wait):
                return None
            self.next_seq += 1
            if self.random.random() < self.loss:
                return None
            values = {k: 0 for k in LEGACY}
            values.update(
                type="CSI_DATA",
                id=seq % 2**32,
                mac="1a:00:00:00:00:00",
                rssi=-43 + self.random.randint(-3, 3),
                rate=11,
                sig_mode=1,
                bandwidth=1,
                channel=11,
                local_timestamp=int(seq * 1e6 / self.rate) % 2**32,
                noise_floor=-95,
                len=128,
                sig_len=40,
                first_word=int(seq == 0),
            )
            values["data"] = (
                '"['
                + ",".join(
                    str(int(24 * math.sin(seq / 30 + i / 8))) for i in range(128)
                )
                + ']"'
            )
            return (
                ",".join(str(values[k]) for k in LEGACY).encode(),
                time.monotonic_ns(),
            )
        chunk = self.handle.read_until(b"\n", size=DEFAULTS["max_serial_line_bytes"])
        timestamp = time.monotonic_ns()
        if not chunk:
            return None
        if self.discarding:
            if chunk.endswith(b"\n"):
                self.discarding = False
            return None
        self.buffer.extend(chunk)
        if len(self.buffer) > DEFAULTS["max_serial_line_bytes"]:
            bad = bytes(self.buffer)
            self.buffer.clear()
            self.discarding = not chunk.endswith(b"\n")
            return bad, timestamp
        if not chunk.endswith(b"\n"):
            return None
        line = bytes(self.buffer)
        self.buffer.clear()
        return line.rstrip(b"\r\n"), timestamp

    def close(self):
        if self.handle:
            self.handle.close()


class CameraSource:
    def __new__(cls, config):
        if config.device.startswith("phone://"):
            return PhoneCameraSource(config)
        return super().__new__(cls)

    def __init__(self, config):
        self.config = config
        self.synthetic = MODE == "synthetic"
        if self.synthetic != (config.device == "synthetic://camera"):
            raise ValueError("Camera device does not match hardware mode")
        self.handle = None
        self.index = 0
        self.next_frame = time.monotonic()
        if not self.synthetic:
            if config.device not in {x["device"] for x in cameras()}:
                raise ValueError("Camera not discovered; reconnect camera and refresh")
            self.handle = cv2.VideoCapture(config.device, cv2.CAP_V4L2)
            try:
                if not self.handle.isOpened():
                    raise RuntimeError(
                        f"Cannot open {config.device}; check device permissions and other camera apps"
                    )
                self.handle.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self.handle.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
                self.handle.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
                self.handle.set(cv2.CAP_PROP_FPS, config.fps)
                self.handle.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                self.close()
                raise

    def read(self, stop):
        if self.synthetic:
            wait = self.next_frame - time.monotonic()
            if wait > 0 and stop.wait(wait):
                return None
            self.next_frame = max(
                self.next_frame + 1 / self.config.fps, time.monotonic()
            )
            frame = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
            frame[:] = (29, 23, 16)
            stamp = time.monotonic_ns()
            cv2.putText(
                frame,
                "SYNTHETIC CAMERA",
                (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (102, 222, 170),
                2,
            )
            cv2.putText(
                frame,
                f"frame {self.index}  monotonic {stamp}",
                (24, 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 230, 230),
                1,
            )
            x = int((self.index % 100) / 100 * (self.config.width - 40)) + 20
            cv2.circle(frame, (x, self.config.height // 2), 14, (102, 222, 170), -1)
            self.index += 1
            return frame, stamp
        ok, frame = self.handle.read()
        stamp = time.monotonic_ns()
        if not ok or frame is None:
            raise RuntimeError(
                f"Camera frame read failed on {self.config.device}; camera may be disconnected"
            )
        return frame, stamp

    def close(self):
        if self.handle:
            self.handle.release()
