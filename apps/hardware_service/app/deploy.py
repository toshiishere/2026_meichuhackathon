"""Exclusive live CSI acquisition for deployment, with an independent camera preview.

Every selected receiver is read by its own thread into its own queue; the
training worker time-aligns them and fuses their scores. Packet and camera
timestamps come from the same monotonic clock in this process, which is what
makes `frame(timestamp_ns)` an alignment rather than a guess.
"""

from collections import deque
import threading
import time
import uuid

import cv2
from fastapi import HTTPException
from .csi import parse_csi
from .sources import SerialSource, CameraSource, validate_port

# A preview frame further than this from the requested CSI time is not the
# same moment, so it is withheld instead of shown next to an unrelated pose.
ALIGNMENT_TOLERANCE_NS = 500_000_000
FRAME_HISTORY_SECONDS = 6

ROW_FIELDS = (
    "bandwidth",
    "sig_mode",
    "stbc",
    "len",
    "first_word",
    "data",
    "tx_seq",
    "firmware_layout",
)


class DeployCapture:
    def __init__(self, jobs):
        self.jobs = jobs
        self.guard = threading.Lock()
        self.token = None
        self.stop = threading.Event()
        self.thread = None
        self.rows = {}
        self.dropped = {}
        self.malformed = {}
        self.receiver_errors = {}
        self.frames = deque(maxlen=90)
        self.jpeg = None
        self.camera_stamp = None
        self.error = self.camera_error = None
        self.last_poll = 0
        self.active = False

    def start(self, options):
        self.jobs.acquire()
        sources = {}
        try:
            for receiver in options.receivers:
                discovered = validate_port(receiver.port)
                if receiver.identity and receiver.identity != discovered["identity"]:
                    raise ValueError(
                        "Receiver identity changed; refresh hardware before deployment"
                    )
                sources[receiver.logical_name] = SerialSource(
                    receiver.port, options.baud_rate
                )
            with self.guard:
                self.token = uuid.uuid4().hex
                self.stop = threading.Event()
                self.rows = {name: deque(maxlen=2000) for name in sources}
                self.dropped = {name: 0 for name in sources}
                self.malformed = {name: 0 for name in sources}
                self.receiver_errors = {}
                # Keep a few seconds of frames so a preview can still be aligned
                # with a CSI window whose end is already slightly in the past.
                self.frames = deque(
                    maxlen=max(
                        30,
                        round(
                            (options.camera.fps if options.camera else 0)
                            * FRAME_HISTORY_SECONDS
                        ),
                    )
                )
                self.jpeg = self.camera_stamp = None
                self.error = self.camera_error = None
                self.last_poll = time.monotonic()
                self.active = True
            self.thread = threading.Thread(
                target=self._run, args=(sources, options.camera), daemon=True
            )
            self.thread.start()
            return dict(capture_id=self.token, receivers=list(sources))
        except Exception:
            for source in sources.values():
                source.close()
            self.jobs.lease.release()
            raise

    def _camera(self, config):
        source = None
        try:
            source = CameraSource(config)
            while not self.stop.is_set():
                # No extra pacing here: the source itself runs at the camera's
                # frame rate, and throttling below it coarsens CSI alignment.
                result = source.read(self.stop)
                if not result:
                    continue
                ok, jpeg = cv2.imencode(
                    ".jpg", result[0], [cv2.IMWRITE_JPEG_QUALITY, 75]
                )
                if ok:
                    with self.guard:
                        self.jpeg, self.camera_stamp = jpeg.tobytes(), result[1]
                        self.frames.append((self.jpeg, self.camera_stamp))
        except Exception as error:
            with self.guard:
                self.camera_error = str(error)
        finally:
            if source:
                source.close()

    def _reader(self, name, source):
        """One receiver: its failure is recorded, the remaining links continue."""
        try:
            while not self.stop.is_set():
                result = source.read(self.stop)
                if not result:
                    continue
                line, stamp = result
                try:
                    row = parse_csi(line)
                    if row:
                        # Preserve source metadata needed to validate the RF layout.
                        row = {k: row.get(k) for k in ROW_FIELDS}
                        row["host_timestamp_ns"] = stamp
                        row["receiver"] = name
                        with self.guard:
                            queue = self.rows[name]
                            if len(queue) == queue.maxlen:
                                self.dropped[name] += 1
                            queue.append(row)
                except ValueError:
                    with self.guard:
                        self.malformed[name] += 1
        except Exception as error:
            with self.guard:
                self.receiver_errors[name] = str(error)
        finally:
            source.close()

    def _run(self, sources, camera):
        threads = []
        try:
            if camera:
                threads.append(
                    threading.Thread(target=self._camera, args=(camera,), daemon=True)
                )
            readers = [
                threading.Thread(
                    target=self._reader, args=(name, source), daemon=True, name=name
                )
                for name, source in sources.items()
            ]
            threads.extend(readers)
            for thread in threads:
                thread.start()
            while not self.stop.is_set():
                if time.monotonic() - self.last_poll > 15:
                    raise RuntimeError(
                        "Deployment client disconnected; acquisition released"
                    )
                if readers and not any(thread.is_alive() for thread in readers):
                    with self.guard:
                        detail = "; ".join(
                            f"{name}: {error}"
                            for name, error in self.receiver_errors.items()
                        )
                    raise RuntimeError(
                        f"Every deployment receiver stopped. {detail}".strip()
                    )
                self.stop.wait(0.1)
        except Exception as error:
            with self.guard:
                self.error = str(error)
        finally:
            self.stop.set()
            for thread in threads:
                thread.join()  # Retain ownership until every device is closed.
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
            rows = {}
            for name, queue in self.rows.items():
                rows[name] = list(queue)
                queue.clear()
            return dict(
                rows=rows,
                active=self.active,
                error=self.error,
                receiver_errors=dict(self.receiver_errors),
                dropped=dict(self.dropped),
                malformed=dict(self.malformed),
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
                if abs(frame[1] - timestamp_ns) > ALIGNMENT_TOLERANCE_NS:
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
