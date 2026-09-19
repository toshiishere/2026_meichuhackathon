"""Session model inference using the training feature path and timestamped CSI.

Training hands every receiver's window for one moment to the same single-link
backbone (`receiver_mode: shared single-link backbone; each receiver is an
example`). Deployment therefore runs all selected receivers over one common
window and fuses their class probabilities into a single pose, rather than
picking one link and discarding the rest.
"""

from collections import Counter, deque
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import uuid

import httpx
import numpy as np
from fastapi import HTTPException

from apps.common.schemas import ID
from apps.common.session_lock import session_lock
from .preprocess import (
    normalize_window,
    packet_amplitude,
    packet_rows,
    resample_window,
)
from .timeline import Timeline

ROOT = Path(__file__).resolve().parents[3]
ACTIVE = {"starting", "running", "stopping"}
# A receiver silent for longer than this is dropped from the fused window
# instead of holding the shared clock (and the camera) back.
STALE_SECONDS = 0.5


def checked_file(session, relative):
    path = session / relative
    if (
        not path.is_file()
        or path.is_symlink()
        or not path.resolve().is_relative_to(session.resolve())
    ):
        raise ValueError(f"Missing or unsafe session artifact: {relative}")
    return path


def model_artifacts(session):
    metadata = json.loads(checked_file(session, "train/model.json").read_text())
    if metadata.get("architecture") != "ESP_Fi_ResNet18":
        raise ValueError("Unsupported model architecture")
    relative = metadata.get("model_path", "")
    if not re.fullmatch(r"train/runs/[a-f0-9]{32}/finetuned_resnet18\.pth", relative):
        raise ValueError("Model must reference an immutable training run")
    checkpoint = checked_file(session, relative)
    classes = json.loads(
        checked_file(session, str(Path(relative).parent / "classes.json")).read_text()
    )
    names = classes.get("class_names", [])
    if (
        len(names) < 2
        or any(not isinstance(n, str) or not n for n in names)
        or len(set(names)) != len(names)
        or classes.get("class_to_idx") != {n: i for i, n in enumerate(names)}
    ):
        raise ValueError("Invalid model class mapping")
    prep = metadata["preprocessing"]
    window, rate = float(prep["window_seconds"]), int(prep["sample_rate_hz"])
    overlap = float(prep["overlap"])
    if not (0.5 <= window <= 10 and 20 <= rate <= 200 and 0 <= overlap < 1):
        raise ValueError("Unsupported model window settings")
    if (
        prep.get("normalization")
        != "per-window global z-score; applied by training loader"
    ):
        raise ValueError("Unsupported model normalization")
    return metadata, checkpoint, names


def replay_files(session):
    files = {}
    for path in sorted((session / "raw").glob("csi_*.csv*")):
        name = path.name.removeprefix("csi_").removesuffix(".zst").removesuffix(".csv")
        if re.fullmatch(ID, name) and path.name.endswith((".csv", ".csv.zst")):
            files[name] = checked_file(session, str(path.relative_to(session)))
    return files


def fuse_scores(names, probabilities, classes):
    """One pose from every receiver: mean of the per-receiver class scores.

    Each receiver observes the same moment through the same trained backbone,
    so its softmax vector is one opinion about that moment; averaging them is
    the deployment counterpart of training on every receiver's windows.
    """
    mean = probabilities.mean(axis=0)
    index = int(mean.argmax())
    return dict(
        label=classes[index],
        confidence=float(mean[index]),
        scores={label: float(score) for label, score in zip(classes, mean)},
        receivers=[
            dict(
                receiver=name,
                label=classes[int(scores.argmax())],
                confidence=float(scores.max()),
                scores={label: float(score) for label, score in zip(classes, scores)},
            )
            for name, scores in zip(names, probabilities)
        ],
        fused_receivers=list(names),
    )


class WindowBuffer:
    def __init__(self, window_seconds, sample_rate_hz):
        self.duration = round(window_seconds * 1e9)
        self.size = round(window_seconds * sample_rate_hz)
        self.rows = deque(maxlen=max(4000, self.size * 4))
        self.accepted = 0
        self.rejected = Counter()
        self.last_stamp = None

    def add(self, row):
        try:
            stamp = int(row["host_timestamp_ns"])
            amp = packet_amplitude(row)
            if self.last_stamp is not None and stamp <= self.last_stamp:
                self.rejected["duplicate_or_out_of_order"] += 1
                return False
            self.last_stamp = stamp
            self.rows.append((stamp, amp))
            self.accepted += 1
            # A second of slack: a receiver running ahead of the shared window
            # end must keep the rows that window still needs.
            cutoff = stamp - self.duration - 1_000_000_000
            while len(self.rows) > 2 and self.rows[1][0] < cutoff:
                self.rows.popleft()
            return True
        except (KeyError, ValueError, TypeError) as error:
            reason = str(error)
            self.rejected[
                (
                    reason
                    if reason in {"unsupported_layout", "invalid_first_word"}
                    else "malformed"
                )
            ] += 1
            return False

    def window(self, end=None):
        """The window ending at `end` on the shared clock, or None if uncovered."""
        if not self.rows:
            return None
        end = self.last_stamp if end is None else end
        stamps, amps = zip(*self.rows)
        return resample_window(stamps, amps, end - self.duration, end, self.size)


class Predictor:
    def __init__(self, metadata, checkpoint, classes):
        import torch

        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("Deployment requires an available ROCm GPU")
        with checkpoint.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != metadata.get(
                "model_sha256"
            ):
                raise ValueError(
                    "Model checksum mismatch; select an intact completed training run"
                )
        sys.path.insert(0, str(ROOT / "csi_model/Model Code"))
        sys.path.insert(0, str(ROOT / "csi_model/finetune/session_tools"))
        from ESP_Fi_model import ESP_Fi_ResNet18

        self.torch, self.normalize, self.classes = torch, normalize_window, classes
        self.model = ESP_Fi_ResNet18(num_classes=len(classes))
        self.model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        try:
            self.model.to("cuda:0").eval()
            # Compile/warm GPU kernels before taking ownership of live hardware.
            prep = metadata["preprocessing"]
            self.scores(
                [
                    np.zeros(
                        (round(prep["window_seconds"] * prep["sample_rate_hz"]), 52),
                        dtype=np.float32,
                    )
                ]
            )
        except Exception:
            self.close()
            raise

    def scores(self, windows):
        """One batched forward over every receiver's window for this moment."""
        torch = self.torch
        began = time.monotonic()
        batch = np.stack([self.normalize(window) for window in windows])[:, None]
        x = torch.from_numpy(np.ascontiguousarray(batch)).to("cuda:0")
        with torch.inference_mode():
            probabilities = self.model(x).softmax(dim=1).cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Model produced non-finite scores")
        return probabilities, (time.monotonic() - began) * 1000

    def close(self):
        self.model = None
        self.torch.cuda.empty_cache()


class Deployment:
    def __init__(self, root, gpu_guard, resolve_session):
        self.root, self.gpu_guard, self.resolve_session = (
            root,
            gpu_guard,
            resolve_session,
        )
        self.guard = threading.Lock()
        self.state = dict(status="idle", prediction=None)
        self.stop_event = threading.Event()
        self.thread = None

    def update(self, **values):
        with self.guard:
            self.state.update(values)

    def snapshot(self):
        with self.guard:
            return json.loads(json.dumps(self.state))

    def catalog(self):
        models, sources, errors = [], [], []
        for path in sorted((self.root / "sessions").glob("*")):
            if not re.fullmatch(ID, path.name):
                continue
            try:
                session = self.resolve_session(self.root, path.name)
                info = json.loads(checked_file(session, "metadata.json").read_text())
                if info.get("status") != "complete":
                    continue
                receivers = list(replay_files(session))
                if receivers:
                    sources.append(dict(session_id=path.name, receivers=receivers))
                if (session / "train/model.json").is_file():
                    meta, _, classes = model_artifacts(session)
                    models.append(
                        dict(
                            session_id=path.name,
                            run_id=meta["run_id"],
                            classes=classes,
                            preprocessing=meta["preprocessing"],
                        )
                    )
            except (ValueError, KeyError, OSError, HTTPException) as error:
                errors.append(dict(session_id=path.name, error=str(error)))
        return dict(models=models, sources=sources, errors=errors)

    def start(self, options):
        if not self.gpu_guard.acquire(blocking=False):
            raise HTTPException(
                409, "GPU is busy training or deploying; stop the active job first"
            )
        leases = ExitStack()
        try:
            sessions = {options.model_session_id}
            if options.source == "replay":
                sessions.add(options.replay_session_id)
            for sid in sorted(sessions):
                leases.enter_context(session_lock(self.root, sid))
                session = self.resolve_session(self.root, sid)
                if (
                    json.loads(checked_file(session, "metadata.json").read_text()).get(
                        "status"
                    )
                    != "complete"
                ):
                    raise ValueError("Choose completed sessions for deployment")
            session = self.resolve_session(self.root, options.model_session_id)
            metadata, checkpoint, classes = model_artifacts(session)
            replays = {}
            if options.source == "replay":
                available = replay_files(
                    self.resolve_session(self.root, options.replay_session_id)
                )
                missing = [r for r in options.replay_receivers if r not in available]
                if missing:
                    raise ValueError(
                        f"Replay receivers not found in the selected session: {', '.join(missing)}"
                    )
                replays = {r: available[r] for r in options.replay_receivers}
                names = list(replays)
            else:
                names = [r.logical_name for r in options.receivers]
            self.stop_event = threading.Event()
            with self.guard:
                self.state = dict(
                    id=uuid.uuid4().hex,
                    status="starting",
                    signal="warming_up",
                    options=options.model_dump(),
                    model_run_id=metadata["run_id"],
                    classes=classes,
                    preprocessing=metadata["preprocessing"],
                    prediction=None,
                    history=[],
                    receiver_names=names,
                    receiver_stats=[],
                    accepted=0,
                    rejected={},
                    source_elapsed_s=0,
                    error=None,
                    capture_id=None,
                    camera_error=None,
                )
            self.thread = threading.Thread(
                target=self._run,
                args=(options, metadata, checkpoint, classes, replays, leases),
                daemon=True,
            )
            self.thread.start()
            return self.snapshot()
        except Exception:
            leases.close()
            self.gpu_guard.release()
            raise

    def stop(self):
        with self.guard:
            if self.state["status"] in ACTIVE:
                self.state["status"] = "stopping"
                self.stop_event.set()
        return self.snapshot()

    def close(self):
        self.stop()
        if self.thread:
            self.thread.join(timeout=15)

    def _run(self, options, metadata, checkpoint, classes, replays, leases):
        capture_id, predictor = None, None
        client = None
        final_status, failure = "stopped", None
        try:
            client = httpx.Client(
                base_url=os.getenv("HARDWARE_URL", "http://hardware-service:8001"),
                timeout=10,
            )
            timeline = None
            if replays:
                try:
                    session = self.resolve_session(self.root, options.replay_session_id)
                    checked_file(session, "raw/video.mp4")
                    timeline = Timeline(
                        checked_file(session, "raw/video_frames.parquet")
                    )
                    self.update(
                        video_available=True, video_time_s=float(timeline.pts[0])
                    )
                except (ValueError, OSError, KeyError) as error:
                    self.update(video_available=False, video_error=str(error))
            predictor = Predictor(metadata, checkpoint, classes)
            if self.stop_event.is_set():
                return
            prep = metadata["preprocessing"]
            names = (
                list(replays)
                if options.source == "replay"
                else [r.logical_name for r in options.receivers]
            )
            buffers = {
                name: WindowBuffer(prep["window_seconds"], prep["sample_rate_hz"])
                for name in names
            }
            duration = buffers[names[0]].duration
            stride = max(100_000_000, round(duration * (1 - prep["overlap"])))
            if options.source == "live" or options.camera:
                response = client.post(
                    "/deploy/capture",
                    json=dict(
                        receivers=(
                            [r.model_dump() for r in options.receivers]
                            if options.source == "live"
                            else []
                        ),
                        camera=options.camera.model_dump() if options.camera else None,
                        baud_rate=options.baud_rate,
                    ),
                )
                if response.is_error:
                    try:
                        detail = response.json().get("detail", response.text)
                    except ValueError:
                        detail = response.text
                    raise RuntimeError(
                        f"Live capture could not start (HTTP {response.status_code}): {detail}"
                    )
                capture_id = response.json()["capture_id"]
                self.update(capture_id=capture_id)
            self.update(status="running")
            origin, last_predict = None, None
            last_receive = {name: time.monotonic() for name in names}
            last_poll = 0
            replay_start = time.monotonic()
            replay_origin = None
            # Peek every replay file so all receivers share one replay origin:
            # the recording's own clock stays the common window clock.
            rows, pending, done = {}, {}, {}
            for name, path in replays.items():
                rows[name] = packet_rows(path)
                pending[name] = next(rows[name], None)
                done[name] = pending[name] is None
                try:
                    stamp = int(pending[name]["host_timestamp_ns"])
                    replay_origin = (
                        stamp if replay_origin is None else min(replay_origin, stamp)
                    )
                except (KeyError, TypeError, ValueError):
                    pass
            ended = False
            while not self.stop_event.is_set():
                incoming = {name: [] for name in names}
                now = time.monotonic()
                if capture_id and now - last_poll >= 0.1:
                    response = client.get(f"/deploy/capture/{capture_id}/packets")
                    response.raise_for_status()
                    batch = response.json()
                    if batch.get("error") or not batch["active"]:
                        raise RuntimeError(
                            batch.get("error") or "Hardware acquisition stopped"
                        )
                    self.update(
                        camera_error=batch.get("camera_error"),
                        receiver_errors=batch.get("receiver_errors") or {},
                        transport_dropped=sum(batch["dropped"].values()),
                        malformed_packets=sum(batch["malformed"].values()),
                    )
                    if options.source == "live":
                        for name, batch_rows in batch["rows"].items():
                            if name in incoming:
                                incoming[name].extend(batch_rows)
                    last_poll = now
                for name, iterator in rows.items():
                    # Pace by original host timestamps, including gaps; memory stays bounded.
                    for _ in range(500):
                        if pending[name] is None:
                            pending[name] = next(iterator, None)
                            if pending[name] is None:
                                done[name] = True
                                break
                        try:
                            stamp = int(pending[name]["host_timestamp_ns"])
                        except (KeyError, ValueError, TypeError):
                            buffers[name].rejected["malformed"] += 1
                            pending[name] = None
                            continue
                        if replay_origin is None:
                            replay_origin = stamp
                        due = (
                            replay_start
                            + (stamp - replay_origin) / 1e9 / options.replay_speed
                        )
                        if due > now:
                            break
                        incoming[name].append(pending[name])
                        pending[name] = None
                ended = bool(rows) and all(done.values())
                for name, batch_rows in incoming.items():
                    for row in batch_rows:
                        if buffers[name].add(row):
                            last_receive[name] = now
                            if origin is None:
                                origin = buffers[name].last_stamp
                if timeline is not None and replay_origin is not None:
                    # Use the paced replay clock, not the last packet/prediction: video
                    # must continue through CSI gaps and share the recording's clock.
                    playhead = replay_origin + round(
                        (now - replay_start) * options.replay_speed * 1e9
                    )
                    stamps = [b.last_stamp for b in buffers.values() if b.last_stamp]
                    if ended and stamps:
                        playhead = max(stamps)
                    self.update(
                        video_time_s=timeline.video_seconds(playhead),
                        video_playing=bool(
                            timeline.ns[0] <= playhead < timeline.ns[-1]
                        ),
                    )
                stale = STALE_SECONDS / (options.replay_speed if replays else 1)
                live = [
                    name
                    for name in names
                    if buffers[name].last_stamp is not None
                    and now - last_receive[name] <= stale
                ]
                rejected = Counter()
                for buffer in buffers.values():
                    rejected.update(buffer.rejected)
                self.update(
                    accepted=sum(b.accepted for b in buffers.values()),
                    rejected=dict(rejected),
                    receiver_stats=[
                        dict(
                            receiver=name,
                            accepted=buffers[name].accepted,
                            rejected=dict(buffers[name].rejected),
                            live=name in live,
                            last_timestamp_ns=buffers[name].last_stamp,
                        )
                        for name in names
                    ],
                )
                if any(b.last_stamp is not None for b in buffers.values()):
                    # One window end shared by every live receiver: the fused
                    # pose, the elapsed clock and the camera all use this time.
                    end = (
                        min(buffers[name].last_stamp for name in live)
                        if live
                        else max(b.last_stamp for b in buffers.values() if b.last_stamp)
                    )
                    elapsed = (end - origin) / 1e9
                    self.update(
                        source_elapsed_s=elapsed,
                        source_timestamp_ns=end,
                    )
                    # Never keep presenting a past prediction as a current pose during a gap.
                    if not live:
                        self.update(signal="no_data", prediction=None)
                    elif (
                        last_predict is None
                        or end - last_predict >= stride
                        or (ended and end != last_predict)
                    ):
                        fused, windows, uncovered = [], [], []
                        for name in live:
                            amplitudes = buffers[name].window(end)
                            if amplitudes is None:
                                uncovered.append(name)
                            else:
                                fused.append(name)
                                windows.append(amplitudes)
                        if not windows:
                            signal = (
                                "warming_up"
                                if elapsed < prep["window_seconds"]
                                else "insufficient_data"
                            )
                            self.update(signal=signal, prediction=None)
                        else:
                            probabilities, inference_ms = predictor.scores(windows)
                            prediction = fuse_scores(fused, probabilities, classes)
                            prediction.update(
                                window_start_ns=end - duration,
                                window_end_ns=end,
                                source_elapsed_s=elapsed,
                                inference_ms=inference_ms,
                                uncovered_receivers=uncovered,
                            )
                            history = self.snapshot()["history"][-29:] + [prediction]
                            self.update(
                                signal="ready", prediction=prediction, history=history
                            )
                            last_predict = end
                else:
                    self.update(signal="waiting_for_supported_csi")
                if ended:
                    if (
                        not any(b.accepted for b in buffers.values())
                        or not self.snapshot()["history"]
                    ):
                        raise ValueError(
                            "Replay contains no usable model windows; check packet layout and coverage"
                        )
                    final_status = "completed"
                    break
                self.stop_event.wait(0.025)
        except Exception as error:
            final_status, failure = "failed", str(error)
        finally:
            if capture_id:
                try:
                    client.post(f"/deploy/capture/{capture_id}/stop")
                except httpx.HTTPError:
                    pass  # Collector's lease expires after 15 seconds without polling.
            try:
                if client is not None:
                    client.close()
                if predictor is not None:
                    predictor.close()
            except Exception as error:
                final_status, failure = "failed", failure or str(error)
            finally:
                try:
                    leases.close()
                finally:
                    self.update(
                        status=final_status,
                        error=failure,
                        capture_id=None,
                        prediction=(
                            None
                            if final_status in {"failed", "stopped"}
                            else self.snapshot().get("prediction")
                        ),
                    )
                    self.gpu_guard.release()
