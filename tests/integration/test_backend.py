import httpx
from fastapi.testclient import TestClient
from apps.backend.app import main
from apps.common.storage import atomic_json


def test_backend_registry_artifacts_and_request_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATA", tmp_path)
    root = tmp_path / "sessions/session1"
    (root / "raw").mkdir(parents=True)
    atomic_json(
        root / "metadata.json",
        {"schema_version": "1.0", "session_id": "session1", "status": "complete"},
    )
    (root / "raw/csi_rx.csv.zst").write_bytes(b"raw-content")

    async def transport(request):
        if request.url.path == "/serial":
            return httpx.Response(
                200,
                json=[
                    dict(
                        identity="board1",
                        port="synthetic://rx0",
                        device="synthetic://rx0",
                        stable_path=None,
                    )
                ],
            )
        return httpx.Response(200, json={"status": "ok"})

    with TestClient(main.app) as client:
        # Override the network client while preserving backend lifespan initialization.
        connection = httpx.AsyncClient(
            transport=httpx.MockTransport(transport), base_url="http://hardware"
        )
        monkeypatch.setattr(main, "client", connection)
        board = dict(
            identity="board1",
            logical_name="rx_left",
            role="csi_receiver",
            port="synthetic://rx0",
        )
        assert client.post("/api/devices", json=board).status_code == 200
        assert client.get("/api/devices").json()[0]["logical_name"] == "rx_left"
        assert client.get("/api/sessions").json()[0]["session_id"] == "session1"
        main.registry.put(
            "sessions", "removed", {"session_id": "removed", "status": "complete"}
        )
        client.get("/api/sessions")
        assert main.registry.get("sessions", "removed") is None
        for host in ["collector.local", "203.0.113.10"]:
            monkeypatch.setenv("PHONE_HOST", host)
            monkeypatch.setenv("PHONE_HTTPS_PORT", "8443")
            assert (
                client.get("/api/health").json()["phone_origin"]
                == f"https://{host}:8443"
            )
        assert (
            client.get("/api/sessions/session1/files/raw/csi_rx.csv.zst").content
            == b"raw-content"
        )
        assert (
            client.get(
                "/api/sessions/session1/files/%2e%2e/%2e%2e/app/backend.sqlite"
            ).status_code
            == 404
        )
        assert (
            client.post("/api/devices", json={**board, "identity": "stale"}).status_code
            == 400
        )
        assert (
            client.post(
                "/api/manifest/rebuild",
                content="{}",
                headers={"Content-Type": "text/plain"},
            ).status_code
            == 415
        )
        assert (
            client.post(
                "/api/manifest/rebuild",
                json={},
                headers={"Origin": "https://external.example"},
            ).status_code
            == 403
        )
        assert client.post("/api/manifest/rebuild", json={}).json() == {"sessions": 1}
        assert (
            client.post("/api/hardware/arbitrary-command", json={}).status_code == 404
        )
        # A disconnected device can be forgotten without querying hardware or
        # changing historical recordings. Its logical name becomes reusable.
        main.registry.put(
            "devices",
            "disconnected",
            {**board, "identity": "disconnected", "logical_name": "old_rx"},
        )
        before = (root / "metadata.json").read_bytes()
        removed = client.post("/api/devices/remove", json={"identity": "disconnected"})
        assert removed.status_code == 200
        assert removed.json() == {"identity": "disconnected", "removed": True}
        assert main.registry.get("devices", "disconnected") is None
        assert (
            client.post(
                "/api/devices", json={**board, "logical_name": "old_rx"}
            ).status_code
            == 200
        )
        assert (root / "metadata.json").read_bytes() == before
        assert (root / "raw/csi_rx.csv.zst").read_bytes() == b"raw-content"
        assert (
            client.post(
                "/api/devices/remove", json={"identity": "disconnected"}
            ).status_code
            == 404
        )
        assert (
            client.post("/api/devices/remove", json={"identity": ""}).status_code == 422
        )


def test_training_proxy_validates_session_and_operations(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATA", tmp_path)
    root = tmp_path / "sessions/session1"
    root.mkdir(parents=True)
    atomic_json(
        root / "metadata.json",
        dict(schema_version="1.0", session_id="session1", status="complete"),
    )
    calls = []

    async def training(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return {"status": "queued"}

    monkeypatch.setattr(main, "training_request", training)
    with TestClient(main.app) as client:
        assert (
            client.post(
                "/api/train/sessions/session1/start", json={"action": "auto"}
            ).status_code
            == 200
        )
        assert calls[-1][1] == "/sessions/session1/start"
        assert calls[-1][2]["json"]["epochs_frozen"] == 5
        assert (
            client.post(
                "/api/train/sessions/session1/start", json={"action": "shell"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/train/sessions/no-session/start", json={"action": "label"}
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/api/train/sessions/session1/start",
                json={"action": "auto"},
                headers={"Origin": "https://other.example"},
            ).status_code
            == 403
        )
        assert client.get("/api/train/jobs/bad/logs").status_code == 404
        assert (
            client.post("/api/train/jobs/" + "a" * 32 + "/delete", json={}).status_code
            == 404
        )
