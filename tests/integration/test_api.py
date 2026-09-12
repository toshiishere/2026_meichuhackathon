import time
from fastapi.testclient import TestClient
from apps.hardware_service.app import main


def test_preflight_record_stop_and_hardware_lease(config, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATA", tmp_path)
    monkeypatch.setattr(main, "recorder", None)
    monkeypatch.setattr(main, "collection_pending", False)
    monkeypatch.setattr(main, "collection_error", None)
    with TestClient(main.app) as client:
        assert len(client.get("/serial").json()) == 3
        snapshot = client.get(
            "/camera/snapshot",
            params={"device": "synthetic://camera", "width": 320, "height": 240},
        )
        assert snapshot.status_code == 200 and snapshot.content.startswith(b"\xff\xd8")
        response = client.post("/collection/start", json=config.model_dump())
        assert response.status_code == 200, response.text
        jid = response.json()["id"]
        assert (
            client.post("/serial/probe", json={"port": "synthetic://rx0"}).status_code
            == 409
        )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            job = client.get("/jobs/" + jid).json()
            if job["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.1)
        assert job["status"] == "completed", job
        assert client.get("/collection/status").json()["status"] == "complete"
        assert (
            client.post("/collection/start", json=config.model_dump()).status_code
            == 409
        )
        assert (
            "Starting collection" in client.get("/jobs/" + jid + "/logs").json()["logs"]
        )


def test_flash_cannot_fake_success_in_synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATA", tmp_path)
    monkeypatch.setattr(main, "recorder", None)
    with TestClient(main.app) as client:
        result = client.post(
            "/flash",
            json={
                "port": "synthetic://rx0",
                "target": "esp32c3",
                "firmware": "csi-recv",
            },
        )
        jid = result.json()["id"]
        for _ in range(50):
            result = client.get("/jobs/" + jid).json()
            if result["status"] == "failed":
                break
            time.sleep(0.02)
        assert result["status"] == "failed"
        assert "unavailable in synthetic mode" in result["error"]
