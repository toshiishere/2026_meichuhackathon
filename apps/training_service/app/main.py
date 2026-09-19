"""Optional ROCm worker. No host environment, shell commands or Docker socket."""

import hashlib
import importlib.metadata
import fcntl
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from apps.common.config import DATA
from apps.common.schemas import ID, TrainRequest, DeployRequest, NotifyRequest
from .deploy import Deployment
from apps.common.session_lock import session_lock
from apps.common.storage import Registry, atomic_json, utc_now

manager = None
deployment = None


def gpu_info():
    try:
        import torch

        ready = bool(torch.version.hip and torch.cuda.is_available())
        return dict(
            ready=ready,
            rocm=torch.version.hip,
            rocm_sdk=importlib.metadata.version("rocm-sdk-core"),
            torch=torch.__version__,
            device=torch.cuda.get_device_name(0) if ready else None,
            error=(
                None
                if ready
                else "ROCm GPU unavailable; check /dev/kfd and /dev/dri access"
            ),
        )
    except Exception as error:
        return dict(ready=False, error=str(error))


def session_path(root, sid):
    if not re.fullmatch(ID, sid):
        raise HTTPException(400, "Invalid session ID")
    path = root / "sessions" / sid
    if (
        not path.is_dir()
        or path.is_symlink()
        or path.resolve().parent != (root / "sessions").resolve()
    ):
        raise HTTPException(404, "Session not found")
    # Do not follow a train/raw directory symlink outside the selected session.
    for child in ("train", "raw"):
        if (path / child).is_symlink():
            raise HTTPException(400, "Session directories cannot be symlinks")
    return path


class TrainingJobs:
    def __init__(self, root):
        self.root, self.guard = root, threading.Lock()
        self.db = Registry(root / "app/training.sqlite")
        # Exactly one worker owns the GPU/job database, even across containers.
        self.worker_lease = (root / "app/training-worker.lock").open("a")
        fcntl.flock(self.worker_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.current = None
        self.cancel = threading.Event()
        self.thread = None
        for job in self.db.list("jobs"):
            if job["status"] in {"queued", "running"}:
                job.update(
                    status="failed",
                    error="Training worker restarted; partial run retained",
                    finished_at=utc_now(),
                )
                self.db.put("jobs", job["id"], job)

    def submit(self, sid, options):
        if not self.guard.acquire(blocking=False):
            raise HTTPException(
                409, "GPU is busy training or deploying; stop the active job first"
            )
        lease = session_lock(self.root, sid)
        entered = False
        try:
            lease.__enter__()
            entered = True
            path = session_path(self.root, sid)
            metadata = json.loads((path / "metadata.json").read_text())
            if metadata.get("status") != "complete":
                raise HTTPException(409, "Choose a completed recording before training")
            required = [path / "raw/video.mp4", path / "raw/video_frames.parquet"]
            if options.action == "finetune":
                required.append(path / "train/action_results.csv")
            if any(not p.is_file() or p.is_symlink() for p in required):
                raise HTTPException(
                    409, "Required video, frame index or action_results.csv is missing"
                )
            gpu = gpu_info()
            if not gpu["ready"]:
                raise HTTPException(503, gpu["error"])
            jid = uuid.uuid4().hex
            folder = path / "train/runs" / jid
            folder.mkdir(parents=True)
            atomic_json(folder / "options.json", options.model_dump())
            job = dict(
                id=jid,
                session_id=sid,
                action=options.action,
                status="queued",
                created_at=utc_now(),
                error=None,
                run_path=f"train/runs/{jid}",
                gpu=gpu,
            )
            self.db.put("jobs", jid, job)
            self.current = jid
            self.cancel.clear()
        except Exception:
            if entered:
                lease.__exit__(None, None, None)
            self.guard.release()
            raise

        def work():
            process = None
            try:
                job.update(status="running", started_at=utc_now())
                self.db.put("jobs", jid, job)
                with (folder / "job.log").open("w", buffering=1) as log:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "apps.training_service.app.pipeline",
                            "--session",
                            str(path),
                            "--run",
                            str(folder),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    while process.poll() is None:
                        if self.cancel.wait(0.2):
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                            break
                    code = process.wait()
                job["status"] = (
                    "cancelled"
                    if self.cancel.is_set()
                    else "completed"
                    if code == 0
                    else "failed"
                )
                if code and not self.cancel.is_set():
                    with (folder / "job.log").open("rb") as log:
                        log.seek(max(0, (folder / "job.log").stat().st_size - 3000))
                        job["error"] = log.read().decode(errors="replace")
            except Exception as error:
                job.update(status="failed", error=str(error))
            finally:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                job["finished_at"] = utc_now()
                try:
                    self.db.put("jobs", jid, job)
                finally:
                    lease.__exit__(None, None, None)
                    self.current = None
                    self.guard.release()

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()
        return job.copy()

    def close(self):
        self.cancel.set()
        if self.thread:
            self.thread.join(timeout=15)
        self.worker_lease.close()


@asynccontextmanager
async def lifespan(app):
    global manager, deployment
    manager = TrainingJobs(DATA)
    deployment = Deployment(DATA, manager.guard, session_path)
    yield
    deployment.close()
    manager.close()


app = FastAPI(title="Session training worker", lifespan=lifespan)


@app.get("/health")
def health():
    return dict(
        status="ok",
        gpu=gpu_info(),
        active_job=manager.current,
        deployment=deployment.snapshot()["status"] if deployment else "idle",
    )


@app.get("/deploy/catalog")
def deployment_catalog():
    return deployment.catalog()


@app.get("/deploy/status")
def deployment_status():
    return deployment.snapshot()


@app.post("/deploy/start")
def deployment_start(options: DeployRequest):
    try:
        return deployment.start(options)
    except (ValueError, KeyError, OSError) as error:
        raise HTTPException(400, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(409, str(error)) from error


@app.post("/deploy/stop")
def deployment_stop():
    return deployment.stop()


@app.post("/deploy/notify")
def deployment_notify(body: NotifyRequest):
    return deployment.set_notify(body.enabled)


@app.get("/sessions/{sid}")
def status(sid: str):
    path = session_path(DATA, sid)
    jobs = [j for j in manager.db.list("jobs") if j["session_id"] == sid]
    for job in jobs:
        progress = path / job["run_path"] / "progress.json"
        if progress.is_file():
            job["progress"] = json.loads(progress.read_text())
    files = [
        dict(path=str(p.relative_to(path)), bytes=p.stat().st_size)
        for p in sorted((path / "train").glob("*"))
        if p.is_file() and not p.name.startswith(".")
    ]
    model_path = path / "train/model.json"
    model = json.loads(model_path.read_text()) if model_path.is_file() else None
    labels_path = path / "train/action_results.csv"
    labels_hash = None
    if labels_path.is_file():
        with labels_path.open("rb") as f:
            labels_hash = hashlib.file_digest(f, "sha256").hexdigest()
    return dict(
        session_id=sid,
        jobs=jobs,
        artifacts=files,
        model_stale=bool(model and model.get("labels_sha256") != labels_hash),
        labels_ready=(path / "train/action_results.csv").is_file(),
        model=model,
    )


@app.post("/sessions/{sid}/start")
def start(sid: str, options: TrainRequest):
    try:
        return manager.submit(sid, options)
    except RuntimeError as error:
        raise HTTPException(409, str(error)) from error


@app.post("/sessions/{sid}/remove/{artifact}")
def remove_artifact(sid: str, artifact: str):
    if artifact not in {"labels", "model"}:
        raise HTTPException(404, "Unknown training artifact")
    session_path(DATA, sid)
    if not manager.guard.acquire(blocking=False):
        raise HTTPException(
            409, "Stop training or deployment before deleting artifacts"
        )
    try:
        with session_lock(DATA, sid):
            path = session_path(DATA, sid)
            train = path / "train"
            names = (
                ("action_results.csv", "action_results.json")
                if artifact == "labels"
                else (
                    "model.json",
                    "finetuned_resnet18.pth",
                    "pretrained_resnet18.pth",
                    "classes.json",
                    "metrics.json",
                    "finetuned_resnet18.onnx",
                    "finetuned_resnet18.int8.onnx",
                    "finetuned_resnet18.int8.json",
                    "finetuned_resnet18.npu.lock",
                )
            )
            # Remove published copies AND historical outputs, preserving raw data,
            # job logs/options and labels_used.csv (model provenance).
            folders = [train] + list((train / "runs").glob("*"))
            if (train / "runs").is_symlink():
                raise HTTPException(400, "Training directories cannot be symlinks")
            files = []
            for folder in folders:
                if folder.is_symlink() or not folder.resolve().is_relative_to(
                    path.resolve()
                ):
                    raise HTTPException(400, "Unsafe training directory")
                for name in names:
                    for filename in (name, f".{name}.tmp"):
                        candidate = folder / filename
                        if candidate.is_symlink():
                            raise HTTPException(
                                400, "Training artifacts cannot be symlinks"
                            )
                        if candidate.is_file():
                            files.append(candidate)
            directories = []
            if artifact == "model":
                for folder in folders:
                    candidate = folder / "finetuned_resnet18.vaip"
                    if candidate.is_symlink():
                        raise HTTPException(400, "NPU cache cannot be a symlink")
                    if candidate.is_dir():
                        directories.append(candidate)
            for candidate in files:
                candidate.unlink()
            for candidate in directories:
                shutil.rmtree(candidate)
            return dict(
                session_id=sid,
                removed=artifact,
                files=len(files) + len(directories),
            )
    except RuntimeError as error:
        raise HTTPException(409, str(error)) from error
    finally:
        manager.guard.release()


@app.get("/jobs/{jid}/logs")
def logs(jid: str):
    job = manager.db.get("jobs", jid)
    if not job:
        raise HTTPException(404, "Training job not found")
    path = session_path(DATA, job["session_id"]) / job["run_path"] / "job.log"
    if not path.is_file():
        return dict(logs="")
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 60000))
        return dict(logs=f.read().decode(errors="replace"))


@app.post("/jobs/{jid}/cancel")
def cancel(jid: str):
    if not manager.db.get("jobs", jid):
        raise HTTPException(404, "Training job not found")
    if manager.current == jid:
        manager.cancel.set()
    return dict(cancel_requested=manager.current == jid)
