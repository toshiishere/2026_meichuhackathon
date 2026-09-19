"""Exclusive live CSI acquisition for deployment, with an independent camera preview."""

from collections import deque
import threading
import time
import uuid

import cv2
from fastapi import HTTPException
from .csi import parse_csi
from .sources import SerialSource, CameraSource, validate_port


class DeployCapture:
    def __init__(self, jobs):
        self.jobs = jobs
        self.guard = threading.Lock()
        self.token = None
        self.stop = threading.Event()
        self.thread = None
        self.rows = deque(maxlen=2000)
        self.frames = deque(maxlen=90)
        self.jpeg = None
        self.camera_stamp = None
        self.error = self.camera_error = None
        self.dropped = self.malformed = 0
        self.last_poll = 0
        self.active = False

    def start(self, options):
        self.jobs.acquire()
        source = None
        try:
            if options.receiver:
                discovered = validate_port(options.receiver.port)
                if (
                    options.receiver.identity
                    and options.receiver.identity != discovered["identity"]
                ):
                    raise ValueError(
                        "Receiver identity changed; refresh hardware before deployment"
                    )
                source = SerialSource(options.receiver.port, options.baud_rate)
            with self.guard:
                self.token = uuid.uuid4().hex
                self.stop = threading.Event()
                self.rows.clear()
                self.frames.clear()
                self.jpeg = self.camera_stamp = None
                self.error = self.camera_error = None
                self.dropped = self.malformed = 0
                self.last_poll = time.monotonic()
                self.active = True
            self.thread = threading.Thread(
                target=self._run, args=(source, options.camera), daemon=True
            )
            self.thread.start()
            return dict(capture_id=self.token)
        except Exception:
            if source:
                source.close()
            self.jobs.lease.release()
            raise

    def _camera(self, config):
        source = None
        try:
            source = CameraSource(config)
            while not self.stop.is_set():
                result = source.read(self.stop)
                if result:
                    ok, jpeg = cv2.imencode(
                        ".jpg", result[0], [cv2.IMWRITE_JPEG_QUALITY, 75]
                    )
                    if ok:
                        with self.guard:
                            self.jpeg, self.camera_stamp = jpeg.tobytes(), result[1]
                            self.frames.append((self.jpeg, self.camera_stamp))
                self.stop.wait(0.08)
        except Exception as error:
            with self.guard:
                self.camera_error = str(error)
        finally:
            if source:
                source.close()

    def _run(self, source, camera):
        camera_thread = None
        try:
            if camera:
                camera_thread = threading.Thread(
                    target=self._camera, args=(camera,), daemon=True
                )
                camera_thread.start()
            while not self.stop.is_set():
                if time.monotonic() - self.last_poll > 15:
                    raise RuntimeError(
                        "Deployment client disconnected; acquisition released"
                    )
                if not source:
                    self.stop.wait(0.1)
                    continue
                result = source.read(self.stop)
                if not result:
                    continue
                line, stamp = result
                try:
                    row = parse_csi(line)
                    if row:
                        # Preserve source metadata needed to validate the RF layout.
                        row = {
                            k: row.get(k)
                            for k in (
                                "bandwidth",
                                "sig_mode",
                                "stbc",
                                "len",
                                "first_word",
                                "data",
                                "tx_seq",
                                "firmware_layout",
                            )
                        }
                        row["host_timestamp_ns"] = stamp
                        with self.guard:
                            if len(self.rows) == self.rows.maxlen:
                                self.dropped += 1
                            self.rows.append(row)
                except ValueError:
                    with self.guard:
                        self.malformed += 1
        except Exception as error:
            with self.guard:
                self.error = str(error)
        finally:
            self.stop.set()
            if source:
                source.close()
            if camera_thread:
                camera_thread.join()  # Retain ownership until every device is closed.
            with self.guard:
                self.active = False
            self.jobs.lease.release()

    def _check(self, token):
        if not self.token or token != self.token:
            raise HTTPException(404, "Deployment capture not found")

    def batch(self, token):
        with self.guard:
            self._check(token)
            self.last_poll = time.monotonic()
            rows = list(self.rows)
            self.rows.clear()
            return dict(
                rows=rows,
                active=self.active,
                error=self.error,
                dropped=self.dropped,
                malformed=self.malformed,
                camera_error=self.camera_error,
                camera_timestamp_ns=self.camera_stamp,
            )

    def frame(self, token, timestamp_ns=None):
        with self.guard:
            self._check(token)
            if (
                not self.jpeg
                or self.stop.is_set()
                or time.monotonic_ns() - self.camera_stamp > 2_000_000_000
            ):
                raise HTTPException(
                    404, self.camera_error or "Waiting for camera frame"
                )
            if timestamp_ns is not None:
                frame = min(self.frames, key=lambda item: abs(item[1] - timestamp_ns))
                if abs(frame[1] - timestamp_ns) > 500_000_000:
                    raise HTTPException(404, "No camera frame aligned with current CSI")
                return frame
            return self.jpeg, self.camera_stamp

    def close(self, token=None):
        with self.guard:
            if token is not None:
                self._check(token)
            self.stop.set()
        if self.thread:
            self.thread.join(timeout=3)
        return dict(stopping=bool(self.thread and self.thread.is_alive()))
