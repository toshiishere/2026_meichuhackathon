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
