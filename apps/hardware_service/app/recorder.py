import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import queue
import shutil
import threading
import time
import traceback
from fractions import Fraction

import av
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

from apps.common.config import DEFAULTS, MODE, ROOT, SCHEMA_VERSION
from apps.common.storage import atomic_json, utc_now, rebuild_manifest
from .csi import FIELDS, GAIN, SequenceTracker, parse_csi
from .sources import SerialSource, CameraSource

FRAME_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("frame_idx", pa.int64()),
        ("video_pts", pa.int64()),
        ("video_time_base_num", pa.int64()),
        ("video_time_base_den", pa.int64()),
        ("video_pts_s", pa.float64()),
        ("host_timestamp_ns", pa.int64()),
        ("wall_timestamp_utc", pa.string()),
    ]
)


class Recorder:
    def __init__(self, config, root, stop=None, preflight_result=None):
        self.config, self.root = config, root
        self.stop = stop or threading.Event()
        self.gate = threading.Event()
        self.lock = threading.RLock()
        self.errors = []
        self.state = "starting"
        self.start_ns = self.end_ns = None
        self.first_video_ns = None
        self.path = root / "sessions" / config.session_id
        self.stats = {
            r.logical_name: dict(
                packets=0,
                written=0,
                parse_errors=0,
                non_csi_lines=0,
                queue_drops=0,
                first_timestamp_ns=None,
                last_timestamp_ns=None,
                rssi=None,
                current_rate_hz=0,
                average_rate_hz=0,
            )
            for r in config.receivers
        }
        self.trackers = {r.logical_name: SequenceTracker() for r in config.receivers}
        self.camera_stats = dict(
            frames_acquired=0,
            frames_recorded=0,
            queue_drops=0,
            first_timestamp_ns=None,
            last_timestamp_ns=None,
            resolution=[config.camera.width, config.camera.height],
            actual_fps=0,
        )
        self.preflight_result = preflight_result
        self.writer_done = []
        self.sources = []
        self.threads = []
        self.producers = []
        self.artifacts = []
        self.outputs = []
        self.last_bytes = 0

    def fail(self, where, error):
        with self.lock:
            self.errors.append(f"{where}: {error}")
        self.stop.set()

    def snapshot(self):
        with self.lock:
            elapsed = (
                ((self.end_ns or time.monotonic_ns()) - self.start_ns) / 1e9
                if self.start_ns
                else 0
            )
            rx = [
                {
                    **s,
                    "logical_name": name,
                    **self.trackers[name].snapshot(),
                    "average_rate_hz": s["packets"] / elapsed if elapsed else 0,
                }
                for name, s in self.stats.items()
            ]
            return dict(
                session_id=self.config.session_id,
                status=self.state,
                elapsed_seconds=elapsed,
                receivers=rx,
                camera=dict(self.camera_stats),
                errors=list(self.errors),
                disk_usage_bytes=self.last_bytes,
                mode=MODE,
            )

    def _thread(self, name, function, *args, producer=False):
        def wrapped():
            try:
                function(*args)
            except Exception as e:
                self.fail(name, str(e))
                with (self.path / "logs" / f"{name}.log").open("a") as log:
                    log.write(traceback.format_exc())

        thread = threading.Thread(name=name, target=wrapped, daemon=True)
        thread.start()
        self.threads.append(thread)
        if producer:
            self.producers.append(thread)
        return thread

    def _serial_reader(self, source, receiver, q, done):
        self.gate.wait()
        name = receiver.logical_name
        last_rate_time, rate_count = time.monotonic(), 0
        pending_gain = {}
        try:
            while not self.stop.is_set():
                value = source.read(self.stop)
                if value is None:
                    continue
                raw, stamp = value
                if self.stop.is_set() or stamp < self.start_ns:
                    continue
                wall = utc_now()
                line = raw.decode("utf-8", errors="replace")
                record, rejection = None, None
                try:
                    record = parse_csi(line)
                    if record:
                        record.update(
                            host_timestamp_ns=stamp,
                            wall_timestamp_utc=wall,
                            receiver_id=name,
                            port=receiver.port,
                            **pending_gain,
                        )
                        pending_gain = {}
                    else:
                        gain = GAIN.search(line)
                        pending_gain = (
                            dict(
                                compensate_gain=gain[1],
                                gain_agc=gain[2],
                                gain_fft=gain[3],
                            )
                            if gain
                            else {}
                        )
                except ValueError as e:
                    pending_gain = {}
                    rejection = str(e)
                with self.lock:
                    s = self.stats[name]
                    if record:
                        s["packets"] += 1
                        rate_count += 1
                        s["first_timestamp_ns"] = s["first_timestamp_ns"] or stamp
                        s["last_timestamp_ns"] = stamp
                        s["rssi"] = record["rssi"]
                        self.trackers[name].update(record["tx_seq"])
                    elif rejection:
                        s["parse_errors"] += 1
                    else:
                        s["non_csi_lines"] += 1
                    now = time.monotonic()
                    if now - last_rate_time >= 1:
                        s["current_rate_hz"] = rate_count / (now - last_rate_time)
                        last_rate_time, rate_count = now, 0
                try:
                    q.put_nowait((record, raw, stamp, wall, rejection))
                except queue.Full:
                    with self.lock:
                        self.stats[name]["queue_drops"] += 1
        finally:
            # Preserve a timed-out trailing fragment for inspection on shutdown.
            if source.buffer:
                try:
                    q.put_nowait(
                        (
                            None,
                            bytes(source.buffer),
                            time.monotonic_ns(),
                            utc_now(),
                            "Partial serial line at stop",
                        )
                    )
                except queue.Full:
                    with self.lock:
                        self.stats[name]["queue_drops"] += 1
            done.set()
            source.close()

    def _camera_reader(self, source, q, done):
        self.gate.wait()
        try:
            while not self.stop.is_set():
                value = source.read(self.stop)
                if value is None:
                    continue
                frame, stamp = value
                if self.stop.is_set() or stamp < self.start_ns:
                    continue
                if [frame.shape[1], frame.shape[0]] != [
                    self.config.camera.width,
                    self.config.camera.height,
                ]:
                    raise RuntimeError(
                        "Camera delivered a different resolution during recording"
                    )
                with self.lock:
                    s = self.camera_stats
                    s["frames_acquired"] += 1
                    s["first_timestamp_ns"] = s["first_timestamp_ns"] or stamp
                    s["last_timestamp_ns"] = stamp
                    span = (stamp - s["first_timestamp_ns"]) / 1e9
                    s["actual_fps"] = (s["frames_acquired"] - 1) / span if span else 0
                try:
                    q.put_nowait((frame, stamp, utc_now()))
                except queue.Full:
                    with self.lock:
                        self.camera_stats["queue_drops"] += 1
        finally:
            done.set()
            source.close()

    @staticmethod
    def _items(q, done):
        while not done.is_set() or not q.empty():
            try:
                yield q.get(timeout=0.1)
            except queue.Empty:
                continue

    def _serial_writer(self, receiver, q, done, temp):
        rejects = self.path / "logs" / f"serial_{receiver.logical_name}.jsonl"
        with temp.open("xb") as file, rejects.open("x") as logs:
            compressor = zstd.ZstdCompressor(level=3).stream_writer(file, closefd=False)
            text = io.TextIOWrapper(compressor, encoding="utf-8", newline="")
            writer = csv.DictWriter(text, fieldnames=FIELDS)
            writer.writeheader()
            last_flush = time.monotonic()
            try:
                for record, raw, stamp, wall, rejection in self._items(q, done):
                    if record:
                        writer.writerow(record)
                        with self.lock:
                            self.stats[receiver.logical_name]["written"] += 1
                    else:
                        logs.write(
                            json.dumps(
                                dict(
                                    host_timestamp_ns=stamp,
                                    wall_timestamp_utc=wall,
                                    error=rejection,
                                    raw_base64=base64.b64encode(raw).decode(),
                                    line=raw.decode("utf-8", errors="replace"),
                                )
                            )
                            + "\n"
                        )
                    if time.monotonic() - last_flush >= 1:
                        text.flush()
                        compressor.flush(zstd.FLUSH_FRAME)
                        file.flush()
                        logs.flush()
                        last_flush = time.monotonic()
            finally:
                text.flush()
                compressor.flush(zstd.FLUSH_FRAME)
                text.close()
                file.flush()
                os.fsync(file.fileno())

    def _video_writer(self, q, done, video_temp, frames_temp, ready):
        container = None
        parquet = None
        try:
            container = av.open(
                str(video_temp),
                mode="w",
                format="mp4",
                options={"movflags": "frag_keyframe+empty_moov+default_base_moof"},
            )
            stream = container.add_stream("libx264", rate=self.config.camera.fps)
            stream.width, stream.height = (
                self.config.camera.width,
                self.config.camera.height,
            )
            stream.pix_fmt = "yuv420p"
            stream.time_base = Fraction(1, 1_000_000)
            stream.codec_context.time_base = stream.time_base
            stream.options = {
                "preset": "ultrafast",
                "tune": "zerolatency",
                "crf": "20",
                "bf": "0",
                "g": str(self.config.camera.fps),
            }
            stream.codec_context.open()
            parquet = pq.ParquetWriter(frames_temp, FRAME_SCHEMA, compression="zstd")
            ready.set()
            rows, idx, last_pts = [], 0, -1
            for frame, stamp, wall in self._items(q, done):
                if self.first_video_ns is None:
                    self.first_video_ns = stamp
                pts = max(last_pts + 1, (stamp - self.first_video_ns) // 1000)
                video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                video_frame.pts = pts
                video_frame.time_base = Fraction(1, 1_000_000)
                for packet in stream.encode(video_frame):
                    container.mux(packet)
                rows.append(
                    dict(
                        schema_version=SCHEMA_VERSION,
                        frame_idx=idx,
                        video_pts=pts,
                        video_time_base_num=1,
                        video_time_base_den=1_000_000,
                        video_pts_s=pts / 1e6,
                        host_timestamp_ns=stamp,
                        wall_timestamp_utc=wall,
                    )
                )
                idx, last_pts = idx + 1, pts
                with self.lock:
                    self.camera_stats["frames_recorded"] = idx
                if len(rows) >= self.config.camera.fps:
                    parquet.write_table(pa.Table.from_pylist(rows, schema=FRAME_SCHEMA))
                    rows.clear()
            for packet in stream.encode():
                container.mux(packet)
            if rows:
                parquet.write_table(pa.Table.from_pylist(rows, schema=FRAME_SCHEMA))
        finally:
            if parquet:
                parquet.close()
            if container:
                container.close()
            ready.set()

    def run(self):
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "raw").mkdir()
        (self.path / "logs").mkdir()
        provenance = json.loads((ROOT / "firmware/provenance.json").read_text())
        collector_hash = hashlib.sha256()
        for folder in ["apps/common", "apps/hardware_service", "configs"]:
            for source_file in sorted((ROOT / folder).rglob("*")):
                if source_file.is_file() and source_file.suffix in {".py", ".yaml"}:
                    collector_hash.update(str(source_file.relative_to(ROOT)).encode())
                    collector_hash.update(source_file.read_bytes())
        metadata = dict(
            schema_version=SCHEMA_VERSION,
            session_id=self.config.session_id,
            status="starting",
            created_at=utc_now(),
            hardware_mode=MODE,
            configuration=self.config.model_dump(),
            clock={
                "source": "time.monotonic_ns",
                "csi_timestamp": "complete serial line receipt",
                "camera_timestamp": "camera read completion",
                "video_pts_origin": "first encoded frame",
                "limitations": "USB, serial and camera driver buffering introduce unmeasured latency",
            },
            software={
                **provenance,
                "project_git_commit": os.getenv("PROJECT_GIT_COMMIT", "unknown"),
                "collector_version": "1.0.0",
                "collector_source_sha256": collector_hash.hexdigest(),
                "dependencies": {
                    p: importlib.metadata.version(p)
                    for p in ["av", "pyarrow", "pyserial", "zstandard", "numpy"]
                },
            },
            preflight=self.preflight_result,
        )
        atomic_json(self.path / "metadata.json", metadata)
        try:
            for rx in self.config.receivers:
                q = queue.Queue(DEFAULTS["serial_queue_size"])
                done = threading.Event()
                self.writer_done.append(done)
                source = SerialSource(
                    rx.port,
                    self.config.baud_rate,
                    self.config.expected_rate_hz,
                    self.config.synthetic_loss,
                    self.config.synthetic_seed + len(self.sources),
                )
                self.sources.append(source)
                # Recheck the actual open handle: a USB driver can reset a board
                # on reopen even after its separate preflight test passed.
                ready_deadline = time.monotonic() + DEFAULTS["stall_timeout_seconds"]
                serial_ready = False
                while not self.stop.is_set() and time.monotonic() < ready_deadline:
                    sample = source.read(self.stop)
                    if sample:
                        try:
                            serial_ready = parse_csi(sample[0].decode(errors="replace")) is not None
                        except ValueError:
                            continue
                        if serial_ready:
                            break
                if not serial_ready:
                    raise RuntimeError(f"{rx.logical_name}: no valid CSI after opening the recording handle")
                temp = self.path / "raw" / f".csi_{rx.logical_name}.csv.zst.tmp"
                self.artifacts.append(
                    (temp, temp.with_name(f"csi_{rx.logical_name}.csv.zst"))
                )
                self._thread(
                    f"writer_{rx.logical_name}", self._serial_writer, rx, q, done, temp
                )
                self._thread(
                    f"reader_{rx.logical_name}",
                    self._serial_reader,
                    source,
                    rx,
                    q,
                    done,
                    producer=True,
                )
            camera = CameraSource(self.config.camera)
            self.sources.append(camera)
            # A successful camera read and open encoder are prerequisites to the start gate.
            warmup = camera.read(self.stop)
            if warmup is None:
                raise RuntimeError("Camera readiness cancelled")
            if [warmup[0].shape[1], warmup[0].shape[0]] != [
                self.config.camera.width,
                self.config.camera.height,
            ]:
                raise RuntimeError("Requested camera resolution not available")
            q, done, ready = (
                queue.Queue(DEFAULTS["camera_queue_size"]),
                threading.Event(),
                threading.Event(),
            )
            self.writer_done.append(done)
            video_temp, frames_temp = (
                self.path / "raw/.video.mp4.tmp",
                self.path / "raw/.video_frames.parquet.tmp",
            )
            self.artifacts.extend(
                [
                    (video_temp, self.path / "raw/video.mp4"),
                    (frames_temp, self.path / "raw/video_frames.parquet"),
                ]
            )
            self._thread(
                "writer_camera",
                self._video_writer,
                q,
                done,
                video_temp,
                frames_temp,
                ready,
            )
            if not ready.wait(20) or self.errors:
                raise RuntimeError(
                    "Video encoder failed to initialize; inspect writer_camera.log"
                )
            self._thread(
                "reader_camera", self._camera_reader, camera, q, done, producer=True
            )
            # Discard pre-start serial buffering after all hardware and outputs are ready.
            for source in self.sources:
                if isinstance(source, SerialSource):
                    source.arm(time.monotonic_ns())
            self.start_ns = time.monotonic_ns()
            for source in self.sources:
                if isinstance(source, SerialSource) and source.synthetic:
                    source.arm(self.start_ns)
            self.state = "recording"
            metadata.update(
                status="recording",
                start_timestamp_ns=self.start_ns,
                started_at=utc_now(),
            )
            atomic_json(self.path / "metadata.json", metadata)
            self.gate.set()
            while not self.stop.wait(0.25):
                now = time.monotonic_ns()
                if (
                    self.config.duration_seconds
                    and (now - self.start_ns) / 1e9 >= self.config.duration_seconds
                ):
                    self.stop.set()
                    break
                with self.lock:
                    stamps = [
                        (name, s["last_timestamp_ns"]) for name, s in self.stats.items()
                    ]
                    stamps.append(("camera", self.camera_stats["last_timestamp_ns"]))
                    for name, stamp in stamps:
                        if (now - (stamp or self.start_ns)) / 1e9 > DEFAULTS[
                            "stall_timeout_seconds"
                        ]:
                            self.fail(
                                name,
                                "No data received for 5 seconds; check device connection and sender",
                            )
                if shutil.disk_usage(self.root).free < DEFAULTS["minimum_free_bytes"]:
                    self.fail(
                        "storage", "Free disk space fell below configured minimum"
                    )
                self.last_bytes = sum(
                    p.stat().st_size for p in self.path.rglob("*") if p.is_file()
                )
        except Exception as e:
            self.fail("collection", str(e))
        finally:
            self.state = "stopping"
            self.stop.set()
            self.end_ns = time.monotonic_ns()
            self.gate.set()
            for thread in self.producers:
                thread.join(timeout=10)
                if thread.is_alive():
                    self.fail(
                        thread.name, "Reader did not stop; service restart required"
                    )
            for done in self.writer_done:
                done.set()
            for thread in self.threads:
                if thread not in self.producers:
                    thread.join(timeout=30)
                    if thread.is_alive():
                        self.fail(
                            thread.name,
                            "Writer did not finalize; service restart required",
                        )
            for source in self.sources:
                if not any(t.is_alive() for t in self.producers):
                    source.close()
            if not self.camera_stats["frames_recorded"]:
                self.fail("video", "No frames recorded")
            for name, stats in self.stats.items():
                if not stats["written"]:
                    self.fail(name, "No valid CSI packets written")
            if not self.errors:
                try:
                    self._verify_video()
                    for temp, final in self.artifacts:
                        with temp.open("rb") as f:
                            os.fsync(f.fileno())
                        os.rename(temp, final)
                except Exception as e:
                    self.fail("finalization", str(e))
            self.last_bytes = sum(p.stat().st_size for p in self.path.rglob('*') if p.is_file())
            self.state = "incomplete" if self.errors else "complete"
            snapshot = self.snapshot()
            degraded = (
                any(
                    s["queue_drops"]
                    or s["parse_errors"]
                    or self.trackers[n].gaps
                    or self.trackers[n].backwards
                    for n, s in self.stats.items()
                )
                or self.camera_stats["queue_drops"]
            )
            degraded = degraded or any(
                r["average_rate_hz"]
                < self.config.expected_rate_hz * self.config.min_rate_ratio
                for r in snapshot["receivers"]
            )
            degraded = (
                degraded
                or self.camera_stats["actual_fps"]
                < self.config.camera.fps * self.config.min_rate_ratio
            )
            metadata.update(
                status=self.state,
                quality="incomplete"
                if self.errors
                else "degraded"
                if degraded
                else "good",
                stop_timestamp_ns=self.end_ns,
                finished_at=utc_now(),
                duration_seconds=snapshot["elapsed_seconds"],
                statistics=snapshot,
                first_video_timestamp_ns=self.first_video_ns,
                errors=self.errors,
                files=[
                    dict(path=str(p.relative_to(self.path)), bytes=p.stat().st_size)
                    for p in (self.path / "raw").iterdir()
                    if p.is_file()
                ],
            )
            atomic_json(self.path / "metadata.json", metadata)
            (self.path / "logs/collection.log").write_text(
                json.dumps(snapshot, indent=2) + "\n"
            )
            rebuild_manifest(self.root)
        return metadata

    def _verify_video(self):
        """Check muxed packet timestamps against the timestamp sidecar, streaming."""
        frame_count = pq.read_metadata(
            self.path / "raw/.video_frames.parquet.tmp"
        ).num_rows
        count = 0
        previous = -1
        with av.open(str(self.path / "raw/.video.mp4.tmp")) as video:
            for packet in video.demux(video.streams.video[0]):
                if packet.pts is not None:
                    if packet.pts <= previous:
                        raise RuntimeError("Video timestamps are not increasing")
                    previous = packet.pts
                    count += 1
        if count != frame_count or count != self.camera_stats["frames_recorded"]:
            raise RuntimeError(
                f"Video/frame index mismatch: video={count}, timestamps={frame_count}"
            )
