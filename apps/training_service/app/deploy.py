"""Session model inference using the training feature path and timestamped CSI."""

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
from .preprocess import packet_amplitude, packet_rows, resample_window

ROOT = Path(__file__).resolve().parents[3]
ACTIVE = {"starting", "running", "stopping"}


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
            cutoff = stamp - self.duration - 300_000_000
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

    def window(self):
        if not self.rows:
            return None
        stamps, amps = zip(*self.rows)
        return resample_window(
            stamps, amps, stamps[-1] - self.duration, stamps[-1], self.size
        )


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
        from finetune import normalize_amplitude

        self.torch, self.normalize, self.classes = torch, normalize_amplitude, classes
        self.model = ESP_Fi_ResNet18(num_classes=len(classes))
        self.model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        try:
            self.model.to("cuda:0").eval()
            # Compile/warm GPU kernels before taking ownership of live hardware.
            prep = metadata["preprocessing"]
            self.predict(
                np.zeros(
                    (round(prep["window_seconds"] * prep["sample_rate_hz"]), 52),
                    dtype=np.float32,
                )
            )
        except Exception:
            self.close()
            raise

    def predict(self, amplitudes):
        torch = self.torch
        began = time.monotonic()
        x = torch.from_numpy(self.normalize(amplitudes)[None, None]).to("cuda:0")
        with torch.inference_mode():
            scores = self.model(x).softmax(dim=1)[0].cpu().numpy()
        if not np.isfinite(scores).all():
            raise RuntimeError("Model produced non-finite scores")
        index = int(scores.argmax())
        return dict(
            label=self.classes[index],
            confidence=float(scores[index]),
            scores={label: float(score) for label, score in zip(self.classes, scores)},
            inference_ms=(time.monotonic() - began) * 1000,
        )

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
            replay = None
            if options.source == "replay":
                replay = replay_files(
                    self.resolve_session(self.root, options.replay_session_id)
                ).get(options.replay_receiver)
                if replay is None:
                    raise ValueError(
                        "Replay receiver not found in the selected session"
                    )
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
                    accepted=0,
                    rejected={},
                    source_elapsed_s=0,
                    error=None,
                    capture_id=None,
                    camera_error=None,
                )
            self.thread = threading.Thread(
                target=self._run,
                args=(options, metadata, checkpoint, classes, replay, leases),
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

    def _run(self, options, metadata, checkpoint, classes, replay, leases):
        capture_id, predictor = None, None
        client = None
        final_status, failure = "stopped", None
        try:
            client = httpx.Client(
                base_url=os.getenv("HARDWARE_URL", "http://hardware-service:8001"),
                timeout=10,
            )
            predictor = Predictor(metadata, checkpoint, classes)
            if self.stop_event.is_set():
                return
            prep = metadata["preprocessing"]
            buffer = WindowBuffer(prep["window_seconds"], prep["sample_rate_hz"])
            stride = max(100_000_000, round(buffer.duration * (1 - prep["overlap"])))
            if options.source == "live" or options.camera:
                response = client.post(
                    "/deploy/capture",
                    json=dict(
                        receiver=(
                            options.receiver.model_dump()
                            if options.source == "live"
                            else None
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
            origin, last_predict, last_receive = None, None, time.monotonic()
            last_poll = 0
            replay_start = time.monotonic()
            replay_origin = None
            rows = iter(packet_rows(replay)) if replay else None
            pending = None
            ended = False
            while not self.stop_event.is_set():
                incoming = []
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
                        transport_dropped=batch["dropped"],
                        malformed_packets=batch["malformed"],
                    )
                    if options.source == "live":
                        incoming = batch["rows"]
                    last_poll = now
                if rows:
                    # Pace by original host timestamps, including gaps; memory stays bounded.
                    for _ in range(500):
                        if pending is None:
                            try:
                                pending = next(rows)
                            except StopIteration:
                                ended = True
                                break
                        try:
                            stamp = int(pending["host_timestamp_ns"])
                        except (ValueError, TypeError, KeyError):
                            buffer.rejected["malformed"] += 1
                            pending = None
                            continue
                        if replay_origin is None:
                            replay_origin = stamp
                        due = (
                            replay_start
                            + (stamp - replay_origin) / 1e9 / options.replay_speed
                        )
                        if due > now:
                            break
                        incoming.append(pending)
                        pending = None
                for row in incoming:
                    if buffer.add(row):
                        last_receive = now
                        if origin is None:
                            origin = buffer.last_stamp
                if buffer.last_stamp is not None:
                    elapsed = (buffer.last_stamp - origin) / 1e9
                    self.update(
                        accepted=buffer.accepted,
                        rejected=dict(buffer.rejected),
                        source_elapsed_s=elapsed,
                    )
                    # Never keep presenting a past prediction as a current pose during a gap.
                    if now - last_receive > 0.5 / (
                        options.replay_speed if replay else 1
                    ):
                        self.update(signal="no_data", prediction=None)
                    elif (
                        last_predict is None
                        or buffer.last_stamp - last_predict >= stride
                        or (ended and buffer.last_stamp != last_predict)
                    ):
                        amp = buffer.window()
                        if amp is None:
                            signal = (
                                "warming_up"
                                if elapsed < prep["window_seconds"]
                                else "insufficient_data"
                            )
                            self.update(signal=signal, prediction=None)
                        else:
                            prediction = predictor.predict(amp)
                            prediction.update(
                                window_start_ns=buffer.last_stamp - buffer.duration,
                                window_end_ns=buffer.last_stamp,
                                source_elapsed_s=elapsed,
                            )
                            history = self.snapshot()["history"][-29:] + [prediction]
                            self.update(
                                signal="ready", prediction=prediction, history=history
                            )
                            last_predict = buffer.last_stamp
                else:
                    self.update(
                        rejected=dict(buffer.rejected),
                        signal="waiting_for_supported_csi",
                    )
                if ended:
                    if not buffer.accepted or not self.snapshot()["history"]:
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
