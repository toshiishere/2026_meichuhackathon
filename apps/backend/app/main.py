import asyncio
import json
import os
import re
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse, Response
from apps.common.config import DATA
from apps.common.schemas import (
    Device,
    ID,
    RemoveSessionRequest,
    RemoveDeviceRequest,
    TrainRequest,
    DeployRequest,
    NotifyRequest,
)
from apps.common.storage import Registry, scan_sessions, read_session, rebuild_manifest

HARDWARE = os.getenv("HARDWARE_URL", "http://hardware-service:8001")
registry = None
client = None


@asynccontextmanager
async def lifespan(app):
    global registry, client
    registry = Registry(DATA / "app/backend.sqlite")
    async with httpx.AsyncClient(base_url=HARDWARE, timeout=30) as connection:
        client = connection
        yield


app = FastAPI(title="CSI Collection Lab", lifespan=lifespan)


@app.middleware("http")
async def local_write_requests(request: Request, call_next):
    if request.method == "POST":
        if request.headers.get("content-type", "").split(";")[0] != "application/json":
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=415,
                content={"detail": "Use application/json for write operations"},
            )
        origin = request.headers.get("origin")
        if origin:
            from urllib.parse import urlsplit

            if urlsplit(origin).netloc != request.headers.get("host"):
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=403,
                    content={"detail": "Cross-origin write requests are not allowed"},
                )
    return await call_next(request)


async def hardware(method, path, **kwargs):
    try:
        response = await client.request(method, path, **kwargs)
    except httpx.RequestError as e:
        raise HTTPException(503, f"Hardware service unavailable: {e}") from e
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise HTTPException(response.status_code, detail)
    return response


@app.get("/api/health")
async def health():
    info = (await hardware("GET", "/health")).json()
    host = os.getenv("PHONE_HOST", "")
    port = os.getenv("PHONE_HTTPS_PORT", "8443")
    return {
        "status": "ok",
        "hardware": info,
        "phone_origin": f"https://{host}:{port}" if host else None,
    }


@app.get("/api/devices")
def devices():
    return registry.list("devices")


@app.post("/api/devices")
async def save_device(device: Device):
    discovered = (await hardware("GET", "/serial")).json()
    current = next(
        (
            d
            for d in discovered
            if d["identity"] == device.identity
            and device.port in {d["port"], d["device"], d["stable_path"]}
        ),
        None,
    )
    if not current:
        raise HTTPException(400, "Device identity or port is stale; refresh hardware")
    for d in registry.list("devices"):
        if (
            d["logical_name"] == device.logical_name
            and d["identity"] != device.identity
        ):
            raise HTTPException(409, "Logical name already assigned to another board")
    old = registry.get("devices", device.identity)
    value = device.model_dump()
    value["firmware"] = old.get("firmware", {}) if old else {}
    registry.put("devices", device.identity, value)
    return value


@app.post("/api/devices/remove")
def remove_device(request: RemoveDeviceRequest):
    if registry.get("devices", request.identity) is None:
        raise HTTPException(404, "Device registration not found")
    # Recordings retain their own configuration snapshots; removing a name does
    # not touch those snapshots, open serial connections, or board firmware.
    registry.delete("devices", request.identity)
    return {"identity": request.identity, "removed": True}


@app.get("/api/sessions")
def sessions():
    values = scan_sessions(DATA)
    existing = {s["session_id"] for s in values}
    for old in registry.list("sessions"):
        if old["session_id"] not in existing:
            registry.delete("sessions", old["session_id"])
    for s in values:
        registry.put(
            "sessions",
            s["session_id"],
            {"session_id": s["session_id"], "status": s["status"]},
        )
    return values


def validated_session(sid):
    if not re.fullmatch(ID, sid):
        raise HTTPException(400, "Invalid session ID")
    try:
        return read_session(DATA, sid)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.get("/api/sessions/{sid}")
def session(sid: str):
    value = validated_session(sid)
    path = DATA / "sessions" / sid
    value["artifacts"] = [
        dict(path=str(p.relative_to(path)), bytes=p.stat().st_size)
        for p in path.rglob("*")
        if p.is_file()
    ]
    return value


@app.get("/api/sessions/{sid}/files/{filename:path}")
def session_file(sid: str, filename: str):
    validated_session(sid)
    root = (DATA / "sessions" / sid).resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(404, "Artifact not found")
    return FileResponse(path, media_type="video/mp4" if path.suffix == ".mp4" else None)


@app.post("/api/sessions/{sid}/recover")
async def recover_session(sid: str):
    validated_session(sid)
    return (await hardware("POST", f"/sessions/{sid}/recover", json={})).json()


@app.post("/api/manifest/rebuild")
def manifest():
    return rebuild_manifest(DATA)


# Explicit allowlist: never expose arbitrary upstream requests or shell execution.
ROUTES = {
    ("GET", "serial"): "/serial",
    ("POST", "serial/probe"): "/serial/probe",
    ("POST", "serial/test"): "/serial/test",
    ("GET", "cameras"): "/cameras",
    ("POST", "camera/test"): "/camera/test",
    ("POST", "camera/close"): "/camera/close",
    ("POST", "flash"): "/flash",
    ("POST", "phone/pair"): "/phone/pair",
}


@app.api_route("/api/hardware/{path:path}", methods=["GET", "POST"])
async def proxy_hardware(path: str, request: Request):
    if path in {"camera/snapshot", "camera/stream"} and request.method == "GET":
        if path == "camera/snapshot":
            result = await hardware(
                "GET", "/camera/snapshot", params=request.query_params
            )
            return Response(
                result.content,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store"},
            )
        upstream = client.build_request(
            "GET", "/camera/stream", params=request.query_params
        )
        try:
            response = await client.send(upstream, stream=True)
        except httpx.RequestError as e:
            raise HTTPException(503, str(e)) from e
        if response.is_error:
            body = await response.aread()
            await response.aclose()
            return Response(
                body, status_code=response.status_code, media_type="application/json"
            )

        async def chunks():
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()

        return StreamingResponse(chunks(), media_type=response.headers["content-type"])
    target = ROUTES.get((request.method, path))
    if not target:
        raise HTTPException(404, "Unknown hardware operation")
    kwargs = {"json": await request.json()} if request.method == "POST" else {}
    return (await hardware(request.method, target, **kwargs)).json()


@app.api_route("/api/collection/{action}", methods=["GET", "POST"])
async def collection(action: str, request: Request):
    if (request.method, action) not in {
        ("GET", "status"),
        ("POST", "preflight"),
        ("POST", "start"),
        ("POST", "stop"),
    }:
        raise HTTPException(404, "Unknown collection operation")
    kwargs = {"json": await request.json()} if action in {"start", "preflight"} else {}
    return (await hardware(request.method, f"/collection/{action}", **kwargs)).json()


async def sync_jobs():
    values = (await hardware("GET", "/jobs")).json()
    for job in values:
        result = job.get("result") or {}
        if (
            job["status"] == "completed"
            and job["kind"] == "flash"
            and result.get("flashed")
        ):
            if registry.get("applied_jobs", job["id"]):
                continue
            ports = (await hardware("GET", "/serial")).json()
            current = next(
                (p for p in ports if p["identity"] == result.get("identity")), None
            )
            device = registry.get("devices", current["identity"]) if current else None
            if device:
                device["firmware"] = result
                registry.put("devices", device["identity"], device)
            registry.put("applied_jobs", job["id"], {"applied": True})
    return values


@app.get("/api/jobs")
async def jobs():
    return await sync_jobs()


@app.api_route("/api/jobs/{jid}/{action}", methods=["GET", "POST"])
async def job_action(jid: str, action: str, request: Request):
    if not re.fullmatch(r"[a-f0-9]{32}", jid) or (request.method, action) not in {
        ("GET", "logs"),
        ("POST", "cancel"),
    }:
        raise HTTPException(404, "Unknown job action")
    return (await hardware(request.method, f"/jobs/{jid}/{action}")).json()


@app.get("/api/events")
async def events(request: Request):
    async def stream():
        while not await request.is_disconnected():
            try:
                status, jobs = await asyncio.gather(
                    hardware("GET", "/collection/status"), sync_jobs()
                )
                payload = {"collection": status.json(), "jobs": jobs}
            except HTTPException as e:
                payload = {"error": e.detail}
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/sessions/{sid}/remove")
async def remove_session(sid: str, body: RemoveSessionRequest):
    if not re.fullmatch(ID, sid) or body.confirm_session_id != sid:
        raise HTTPException(400, "Type the exact session ID to confirm removal")
    return (
        await hardware("POST", f"/sessions/{sid}/remove", json=body.model_dump())
    ).json()


# Optional worker: its absence must not interrupt recording or hardware events.
async def training_request(method, path, **kwargs):
    address = os.getenv("TRAINING_URL", "http://training-service:8002")
    try:
        async with httpx.AsyncClient(base_url=address, timeout=15) as training:
            response = await training.request(method, path, **kwargs)
    except httpx.RequestError as error:
        raise HTTPException(
            503,
            "Training service unavailable. Run make up to start the ROCm worker, and check make logs if it remains unavailable.",
        ) from error
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise HTTPException(response.status_code, detail)
    return response.json()


@app.get("/api/train/health")
async def training_health():
    return await training_request("GET", "/health")


@app.get("/api/train/sessions/{sid}")
async def training_status(sid: str):
    validated_session(sid)
    return await training_request("GET", f"/sessions/{sid}")


@app.post("/api/train/sessions/{sid}/start")
async def start_training(sid: str, body: TrainRequest):
    validated_session(sid)
    return await training_request(
        "POST", f"/sessions/{sid}/start", json=body.model_dump()
    )


@app.post("/api/train/sessions/{sid}/remove/{artifact}")
async def remove_training_artifact(sid: str, artifact: str):
    validated_session(sid)
    if artifact not in {"labels", "model"}:
        raise HTTPException(404, "Unknown training artifact")
    return await training_request("POST", f"/sessions/{sid}/remove/{artifact}", json={})


@app.api_route("/api/train/jobs/{jid}/{action}", methods=["GET", "POST"])
async def training_job(jid: str, action: str, request: Request):
    if not re.fullmatch(r"[a-f0-9]{32}", jid) or (request.method, action) not in {
        ("GET", "logs"),
        ("POST", "cancel"),
    }:
        raise HTTPException(404, "Unknown training job action")
    return await training_request(request.method, f"/jobs/{jid}/{action}")


@app.get("/api/deploy/catalog")
async def deploy_catalog():
    return await training_request("GET", "/deploy/catalog")


@app.get("/api/deploy/status")
async def deploy_status():
    return await training_request("GET", "/deploy/status")


@app.post("/api/deploy/start")
async def deploy_start(body: DeployRequest):
    if body.source == "live":
        discovered = (await hardware("GET", "/serial")).json()
        for receiver in body.receivers:
            saved = registry.get("devices", receiver.identity)
            if not saved or saved["role"] != "csi_receiver":
                raise HTTPException(400, "Choose registered CSI receivers")
            port = next(
                (p for p in discovered if p["identity"] == receiver.identity), None
            )
            if not port:
                raise HTTPException(
                    409,
                    f"Receiver {saved['logical_name']} disconnected; refresh hardware",
                )
            receiver.port = port["port"]
            receiver.logical_name = saved["logical_name"]
        names = [r.logical_name for r in body.receivers]
        if len(set(names)) != len(names):
            raise HTTPException(409, "Selected receivers share a logical name")
    return await training_request("POST", "/deploy/start", json=body.model_dump())


@app.post("/api/deploy/stop")
async def deploy_stop():
    return await training_request("POST", "/deploy/stop")


@app.post("/api/deploy/notify")
async def deploy_notify(body: NotifyRequest):
    return await training_request(
        "POST", "/deploy/notify", json=body.model_dump()
    )


@app.get("/api/deploy/camera/{capture_id}")
async def deploy_camera(
    capture_id: str, timestamp_ns: int | None = Query(default=None, ge=0)
):
    if not re.fullmatch(r"[a-f0-9]{32}", capture_id):
        raise HTTPException(404, "Capture not found")
    result = await hardware(
        "GET",
        f"/deploy/capture/{capture_id}/camera",
        params={"timestamp_ns": timestamp_ns} if timestamp_ns is not None else {},
    )
    return Response(
        result.content,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Host-Timestamp-Ns": result.headers.get("X-Host-Timestamp-Ns", ""),
        },
    )
