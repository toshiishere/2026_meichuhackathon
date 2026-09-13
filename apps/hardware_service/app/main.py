import asyncio
import json
import re
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
import cv2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, Response
from apps.common.config import DATA, MODE, DEFAULTS
from apps.common.schemas import (
    CollectionConfig,
    SerialRequest,
    CameraConfig,
    FlashRequest,
    PhonePairRequest,
    RemoveSessionRequest,
    ID,
)
from apps.common.storage import scan_sessions, atomic_json, rebuild_manifest
from .sources import serial_devices, cameras, CameraSource
from .checks import serial_test, camera_test, preflight
from .jobs import Jobs, BusyError
from .firmware import probe, flash
from .recorder import Recorder
from .phone import hub

jobs = None
recorder = None
collection_pending = False
collection_error = None
collection_stop = None
preflight_status = None
preview_token = None
preview_stop = None


@asynccontextmanager
async def lifespan(app):
    global jobs
    DATA.mkdir(parents=True, exist_ok=True)
    jobs = Jobs(DATA)
    for metadata in scan_sessions(DATA):
        if metadata["status"] in {"starting", "recording", "stopping"}:
            metadata.update(
                status="incomplete",
                quality="incomplete",
                errors=[
                    "Collector restarted before finalization; temporary raw files preserved"
                ],
            )
            atomic_json(
                DATA / "sessions" / metadata["session_id"] / "metadata.json", metadata
            )
    rebuild_manifest(DATA)
    yield
    if collection_stop:
        collection_stop.set()
    # Allow the recorder to drain queues before uvicorn exits.
    for _ in range(80):
        if not collection_pending and (
            not recorder or recorder.state in {"complete", "incomplete"}
        ):
            break
        await asyncio.sleep(0.5)


app = FastAPI(title="CSI Lab hardware service", lifespan=lifespan)


@app.exception_handler(BusyError)
async def busy(request, error):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=409, content={"detail": str(error)})


@app.exception_handler(ValueError)
async def invalid(request, error):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=400, content={"detail": str(error)})


@app.get("/health")
def health():
    return {"status": "ok", "mode": MODE, "defaults": DEFAULTS}


@app.get("/serial")
def serial():
    return serial_devices()


@app.get("/cameras")
def camera_list():
    return cameras()


@app.post("/serial/probe")
def serial_probe(body: SerialRequest):
    return jobs.submit("probe", lambda log, stop: probe(body.port, log, stop))


@app.post("/serial/test")
def test_serial(body: SerialRequest):
    return jobs.submit("serial-test", lambda log, stop: serial_test(body, stop))


@app.post("/camera/test")
def test_camera(body: CameraConfig):
    return jobs.submit("camera-test", lambda log, stop: camera_test(body, 5, stop))


@app.get("/camera/snapshot")
def snapshot(
    device: str,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    fixed_frame_rate: bool = True,
):
    config = CameraConfig(
        device=device,
        width=width,
        height=height,
        fps=fps,
        fixed_frame_rate=fixed_frame_rate,
    )
    jobs.acquire()
    try:
        source = CameraSource(config)
        try:
            frame, stamp = source.read(threading.Event())
            ok, data = cv2.imencode(".jpg", frame)
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            return Response(
                data.tobytes(),
                media_type="image/jpeg",
                headers={
                    "X-Host-Timestamp-Ns": str(stamp),
                    "Cache-Control": "no-store",
                },
            )
        finally:
            source.close()
    finally:
        jobs.lease.release()


@app.get("/camera/stream")
async def stream(
    device: str,
    token: str,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    fixed_frame_rate: bool = True,
):
    global preview_token, preview_stop
    import uuid

    uuid.UUID(token)
    config = CameraConfig(
        device=device,
        width=width,
        height=height,
        fps=fps,
        fixed_frame_rate=fixed_frame_rate,
    )
    jobs.acquire()
    try:
        source = await asyncio.to_thread(CameraSource, config)
    except Exception:
        jobs.lease.release()
        raise
    stop = threading.Event()
    preview_token, preview_stop = token, stop

    async def frames():
        global preview_token, preview_stop
        try:
            while not stop.is_set():
                result = await asyncio.to_thread(source.read, stop)
                if result:
                    ok, data = cv2.imencode(".jpg", result[0])
                    if ok:
                        yield (
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                            + data.tobytes()
                            + b"\r\n"
                        )
                await asyncio.sleep(0.1)
        finally:
            stop.set()
            source.close()
            preview_token = preview_stop = None
            jobs.lease.release()

    return StreamingResponse(
        frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/camera/close")
async def close_preview(body: dict):
    if preview_token != body.get("token") or preview_stop is None:
        return {"status": "closed"}
    token = preview_token
    preview_stop.set()
    for _ in range(250):
        if preview_token != token:
            return {"status": "closed"}
        await asyncio.sleep(0.02)
    raise HTTPException(503, "Camera preview did not stop; restart hardware service")


@app.post("/flash")
def flash_firmware(body: FlashRequest):
    return jobs.submit("flash", lambda log, stop: flash(body, log, stop))


@app.post("/collection/preflight")
def check_preflight(body: CollectionConfig):
    def work(log, stop):
        global preflight_status
        preflight_status = preflight(body, DATA, stop)
        log(json.dumps(preflight_status))
        return preflight_status

    return jobs.submit("preflight", work)


@app.post("/collection/start")
def start(body: CollectionConfig):
    global collection_pending, collection_error, recorder
    if (DATA / "sessions" / body.session_id).exists():
        raise HTTPException(
            409, "Session ID already exists; raw recordings cannot be overwritten"
        )
    if recorder and any(t.is_alive() for t in recorder.threads):
        raise BusyError(
            "Previous acquisition threads are still running; restart hardware service before continuing"
        )

    def work(log, stop):
        global recorder, collection_stop, collection_pending, collection_error, preflight_status
        collection_stop = stop
        collection_pending = True
        collection_error = None
        try:
            log("Running mandatory preflight with current hardware configuration")
            preflight_status = preflight(body, DATA, stop)
            log(json.dumps(preflight_status))
            if not preflight_status["passed"]:
                raise RuntimeError(
                    "Preflight failed or was cancelled. Review receiver, camera and disk checks."
                )
            recorder = Recorder(body, DATA, stop, preflight_status)
            collection_pending = False
            metadata = recorder.run()
            log(f"Session {metadata['session_id']}: {metadata['status']}")
            if metadata["status"] != "complete":
                raise RuntimeError("; ".join(metadata["errors"]))
            return metadata
        except Exception as e:
            collection_error = str(e)
            raise
        finally:
            collection_pending = False

    return jobs.submit("collection", work)


@app.post("/collection/stop")
def stop_recording():
    if collection_stop:
        collection_stop.set()
    return {"status": "stop_requested"}


@app.get("/collection/status")
def collection_status():
    status = recorder.snapshot() if recorder else {"status": "idle", "mode": MODE}
    if collection_pending:
        status = {"status": "preflight", "mode": MODE}
    return {**status, "preflight": preflight_status, "last_error": collection_error}


@app.get("/jobs")
def all_jobs():
    return jobs.db.list("jobs")[:100]


@app.get("/jobs/{jid}")
def get_job(jid: str):
    value = jobs.db.get("jobs", jid)
    if not value:
        raise HTTPException(404, "Job not found")
    return value


@app.get("/jobs/{jid}/logs")
def logs(jid: str):
    get_job(jid)
    # Only DB-generated job identifiers are used to construct this path.
    path = DATA / "app/job_logs" / (jid + ".log")
    if not path.exists():
        return {"logs": ""}
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 60000))
        return {"logs": f.read().decode(errors="replace")}


@app.post("/jobs/{jid}/cancel")
def cancel(jid: str):
    get_job(jid)
    return jobs.cancel(jid)


@app.post("/phone/pair")
def pair_phone(body: PhonePairRequest):
    return hub.pair(body.model_dump())


@app.websocket("/phone/stream")
async def phone_stream(socket: WebSocket):
    # The LAN proxy exposes only this token-authenticated upload endpoint.
    await socket.accept()
    pid, connected = None, False
    try:
        auth = await asyncio.wait_for(socket.receive_json(), timeout=15)
        if not isinstance(auth, dict) or not isinstance(auth.get("id"), str):
            raise ValueError("Invalid phone handshake")
        pid = auth["id"]
        settings = hub.settings(pid, auth.get("token"))
        await socket.send_json({"type": "settings", **settings})
        ready = await asyncio.wait_for(socket.receive_json(), timeout=120)
        if not isinstance(ready, dict) or ready.get("type") != "ready":
            raise ValueError("Expected phone camera readiness")
        hub.connect(pid, auth.get("token"), ready.get("native_resolution"))
        connected = True
        await socket.send_json({"type": "ready", "device": f"phone://{pid}"})
        while True:
            payload = await asyncio.wait_for(socket.receive_bytes(), timeout=10)
            stamp = time.monotonic_ns()
            result = await asyncio.to_thread(hub.accept_frame, pid, payload, stamp)
            await socket.send_json(result)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await socket.send_json(
                {
                    "type": "error",
                    "message": str(e) or "Phone stopped responding; pair again",
                }
            )
            await socket.close(code=1008)
        except Exception:
            pass
    finally:
        if connected:
            hub.disconnect(pid)


@app.post("/sessions/{sid}/remove")
def remove_session(sid: str, body: RemoveSessionRequest):
    if not re.fullmatch(ID, sid) or body.confirm_session_id != sid:
        raise HTTPException(400, "Type the exact session ID to confirm removal")
    path = DATA / "sessions" / sid
    if (
        not path.is_dir()
        or path.is_symlink()
        or path.resolve().parent != (DATA / "sessions").resolve()
    ):
        raise HTTPException(404, "Session not found")

    def work(log, stop):
        if (
            recorder
            and recorder.config.session_id == sid
            and (
                recorder.state not in {"complete", "incomplete"}
                or any(t.is_alive() for t in recorder.threads)
            )
        ):
            raise RuntimeError(
                "Cannot remove a recording or a session with active acquisition threads"
            )
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            try:
                state = json.loads(metadata_path.read_text()).get("status")
            except ValueError:
                state = "unreadable"
            if state in {"starting", "recording", "stopping"}:
                raise RuntimeError(
                    "Cannot remove a recording session; stop it or restart the collector to recover it first"
                )
        # First remove the whole directory from the archive atomically. If disk
        # cleanup fails, the partial directory stays outside the canonical archive.
        trash = DATA / "app/removing_sessions"
        trash.mkdir(parents=True, exist_ok=True)
        target = trash / f"{sid}_{uuid.uuid4().hex}"
        path.rename(target)
        log(f"Removing session {sid}: raw CSI, video, timestamps, metadata and logs")
        try:
            shutil.rmtree(target)
        finally:
            rebuild_manifest(DATA)
        return {"session_id": sid, "removed": True}

    return jobs.submit("session-remove", work)
