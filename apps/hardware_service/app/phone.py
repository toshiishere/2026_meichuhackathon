"""Paired browser camera ingest in the same monotonic clock domain as CSI.

Phone performance.now() is preserved, NEVER used as the host synchronization
clock. Frames are stamped on websocket message receipt, before JPEG decoding.
Each upload is acknowledged; the browser allows a bounded window of outstanding
frames so capture does not wait for a network round trip after every JPEG.
"""

import math
import queue
import secrets
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np

from apps.common.config import DEFAULTS

HEADER = struct.Struct(
    "!4sIdI"
)  # magic, frame sequence, phone capture ms, skipped total
MAX_FRAME_BYTES = 2 * 1024 * 1024
PAIR_TTL_SECONDS = 600


@dataclass
class Subscription:
    frames: queue.Queue = field(
        default_factory=lambda: queue.Queue(DEFAULTS["camera_queue_size"])
    )
    dropped: int = 0


@dataclass
class Phone:
    id: str
    token: str
    settings: dict
    expires: float
    connected: bool = False
    consumed: bool = False
    received: int = 0
    last_timestamp_ns: int | None = None
    last_sequence: int = -1
    last_capture_ms: float = -1
    skipped: int = 0
    native_resolution: list = field(default_factory=list)
    subscribers: list = field(default_factory=list)


class PhoneHub:
    def __init__(self):
        self.lock = threading.RLock()
        self.phones = {}

    def pair(self, settings):
        with self.lock:
            now = time.monotonic()
            self.phones = {
                k: v
                for k, v in self.phones.items()
                if v.connected or v.subscribers or (not v.consumed and v.expires > now)
            }
            if len(self.phones) >= 8:
                raise ValueError(
                    "At most eight pending or connected phone cameras are allowed"
                )
            phone = Phone(
                uuid.uuid4().hex,
                secrets.token_urlsafe(32),
                settings,
                now + PAIR_TTL_SECONDS,
            )
            self.phones[phone.id] = phone
            return dict(
                id=phone.id,
                token=phone.token,
                expires_in_seconds=PAIR_TTL_SECONDS,
                **settings,
            )

    def connect(self, pid, token, native_resolution):
        with self.lock:
            phone = self.phones.get(pid)
            if (
                not phone
                or not isinstance(token, str)
                or not secrets.compare_digest(phone.token, token)
            ):
                raise ValueError("Invalid phone pairing link")
            if phone.connected or phone.consumed or phone.expires <= time.monotonic():
                raise ValueError(
                    "Pairing link expired or already used; create a new link on the computer"
                )
            if (
                not isinstance(native_resolution, list)
                or len(native_resolution) != 2
                or any(
                    type(x) is not int or not 1 <= x <= 8192 for x in native_resolution
                )
            ):
                raise ValueError("Invalid phone camera resolution")
            phone.connected = phone.consumed = True
            phone.native_resolution = native_resolution
            return phone.settings

    def disconnect(self, pid):
        with self.lock:
            if pid in self.phones:
                self.phones[pid].connected = False

    def settings(self, pid, token):
        """Read pairing configuration without consuming the one-use credential."""
        with self.lock:
            p = self.phones.get(pid)
            if (
                not p
                or not isinstance(token, str)
                or not secrets.compare_digest(p.token, token)
                or p.consumed
                or p.expires <= time.monotonic()
            ):
                raise ValueError("Invalid, expired, or used pairing link")
            return p.settings

    def accept_frame(self, pid, payload, host_timestamp_ns):
        if not HEADER.size < len(payload) <= MAX_FRAME_BYTES:
            raise ValueError("Phone frame exceeds the 2 MiB limit or is empty")
        magic, sequence, capture_ms, skipped = HEADER.unpack_from(payload)
        if magic != b"CSI1" or not math.isfinite(capture_ms) or capture_ms < 0:
            raise ValueError("Invalid phone frame header")
        with self.lock:
            phone = self.phones[pid]
            if not phone.connected:
                raise ValueError("Phone disconnected")
            if (
                sequence <= phone.last_sequence
                or capture_ms <= phone.last_capture_ms
                or skipped < phone.skipped
            ):
                raise ValueError(
                    "Phone sequence, capture time, and drop counters must increase monotonically"
                )
            settings = phone.settings
        # Decode a bounded JPEG; reject dimensions before allocating a decoded frame.
        encoded = payload[HEADER.size :]
        if jpeg_dimensions(encoded) != (settings["width"], settings["height"]):
            raise ValueError(
                "Phone frame dimensions differ from the paired capture settings"
            )
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[:2] != (settings["height"], settings["width"]):
            raise ValueError("Invalid phone JPEG frame")
        metadata = dict(
            phone_frame_idx=sequence,
            phone_capture_timestamp_ms=capture_ms,
            phone_skipped_frames_total=skipped,
        )
        with self.lock:
            phone.last_sequence = sequence
            phone.last_capture_ms = capture_ms
            phone.last_timestamp_ns = host_timestamp_ns
            phone.skipped = skipped
            phone.received += 1
            for subscriber in phone.subscribers:
                try:
                    subscriber.frames.put_nowait((frame, host_timestamp_ns, metadata))
                except queue.Full:
                    subscriber.dropped += 1
        return dict(type="ack", frame_idx=sequence)

    def discover(self):
        with self.lock:
            return [
                dict(
                    device=f"phone://{p.id}",
                    name=p.settings["name"],
                    width=p.settings["width"],
                    height=p.settings["height"],
                    fps=p.settings["fps"],
                    frames_received=p.received,
                    last_frame_timestamp_ns=p.last_timestamp_ns,
                    native_resolution=p.native_resolution,
                    formats=f"Phone browser: {p.settings['width']} × {p.settings['height']} @ {p.settings['fps']} FPS; host timestamps measure network arrival",
                )
                for p in self.phones.values()
                if p.connected
            ]

    def subscribe(self, config):
        with self.lock:
            phone = self.phones.get(config.device.removeprefix("phone://"))
            if not phone or not phone.connected:
                raise ValueError(
                    "Phone camera is disconnected; keep the phone page open and connect again"
                )
            if any(
                getattr(config, k) != phone.settings[k]
                for k in ("width", "height", "fps")
            ):
                raise ValueError(
                    "Use the resolution and FPS selected when pairing this phone"
                )
            sub = Subscription()
            phone.subscribers.append(sub)
            return phone, sub


# JPEG SOF marker parser: no oversized decompression from an untrusted upload.
def jpeg_dimensions(data):
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("Expected a JPEG image")
    offset = 2
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            raise ValueError("Malformed JPEG marker")
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(data):
            break
        length = int.from_bytes(data[offset : offset + 2], "big")
        if length < 2 or offset + length > len(data):
            break
        if marker in {0xC0, 0xC1, 0xC2}:
            if length < 8:
                break
            return (
                int.from_bytes(data[offset + 5 : offset + 7], "big"),
                int.from_bytes(data[offset + 3 : offset + 5], "big"),
            )
        if marker == 0xDA:
            break
        offset += length
    raise ValueError("Unsupported or truncated JPEG header")


hub = PhoneHub()


class PhoneCameraSource:
    def __init__(self, config):
        self.phone, self.subscription = hub.subscribe(config)
        self.frame_metadata = {}
        self.skipped_at_start = self.phone.skipped
        self.closed = False

    def read(self, stop):
        deadline = time.monotonic() + DEFAULTS["stall_timeout_seconds"]
        while not stop.is_set():
            if not self.phone.connected:
                raise RuntimeError(
                    "Phone camera disconnected; keep the phone awake and its camera page visible"
                )
            try:
                frame, stamp, self.frame_metadata = self.subscription.frames.get(
                    timeout=0.1
                )
                return frame, stamp
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Phone camera stopped sending frames; check Wi-Fi and the phone screen"
                    )
        return None

    @property
    def transport_stats(self):
        return dict(
            phone_queue_drops=self.subscription.dropped,
            phone_skipped_frames=self.phone.skipped - self.skipped_at_start,
        )

    def close(self):
        with hub.lock:
            if not self.closed:
                self.phone.subscribers.remove(self.subscription)
                self.closed = True
