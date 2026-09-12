import time
from types import SimpleNamespace
from fastapi.testclient import TestClient
import pyarrow.parquet as pq
from apps.common.storage import atomic_json
from apps.hardware_service.app import main


def test_session_removal_requires_confirmation_and_cleans_archive(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(main, "DATA", tmp_path)
    monkeypatch.setattr(main, "recorder", None)
    root = tmp_path / "sessions/removable"
    (root / "raw").mkdir(parents=True)
    (root / "raw/video.mp4").write_bytes(b"test-video")
    atomic_json(
        root / "metadata.json",
        {"schema_version": "1.0", "session_id": "removable", "status": "complete"},
    )
    with TestClient(main.app) as client:
        assert (
            client.post(
                "/sessions/removable/remove", json={"confirm_session_id": "wrong"}
            ).status_code
            == 400
        )
        main.jobs.acquire()
        try:
            assert (
                client.post(
                    "/sessions/removable/remove",
                    json={"confirm_session_id": "removable"},
                ).status_code
                == 409
            )
            assert root.exists()
        finally:
            main.jobs.lease.release()
        job = client.post(
            "/sessions/removable/remove", json={"confirm_session_id": "removable"}
        ).json()
        for _ in range(100):
            job = client.get("/jobs/" + job["id"]).json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.02)
        assert job["status"] == "completed", job
        assert job["result"] == {"session_id": "removable", "removed": True}
        assert not root.exists()
        assert pq.read_table(tmp_path / "manifest.parquet").num_rows == 0
        assert (
            client.post(
                "/sessions/removable/remove", json={"confirm_session_id": "removable"}
            ).status_code
            == 404
        )


def test_recording_metadata_cannot_be_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATA", tmp_path)
    monkeypatch.setattr(main, "recorder", None)
    with TestClient(main.app) as client:
        root = tmp_path / "sessions/active"
        root.mkdir(parents=True)
        atomic_json(
            root / "metadata.json",
            {"schema_version": "1.0", "session_id": "active", "status": "recording"},
        )
        job = client.post(
            "/sessions/active/remove", json={"confirm_session_id": "active"}
        ).json()
        for _ in range(100):
            job = client.get("/jobs/" + job["id"]).json()
            if job["status"] == "failed":
                break
            time.sleep(0.02)
        assert job["status"] == "failed" and root.exists()


def test_removal_rejects_symlinks_and_surviving_acquisition_threads(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(main, "DATA", tmp_path)
    monkeypatch.setattr(main, "recorder", None)
    outside = tmp_path / "outside"
    outside.mkdir()
    with TestClient(main.app) as client:
        (tmp_path / "sessions").mkdir(exist_ok=True)
        (tmp_path / "sessions/alias").symlink_to(outside, target_is_directory=True)
        assert (
            client.post(
                "/sessions/alias/remove", json={"confirm_session_id": "alias"}
            ).status_code
            == 404
        )
        assert outside.exists()
        root = tmp_path / "sessions/unfinished"
        root.mkdir()
        atomic_json(
            root / "metadata.json",
            {
                "schema_version": "1.0",
                "session_id": "unfinished",
                "status": "incomplete",
            },
        )
        monkeypatch.setattr(
            main,
            "recorder",
            SimpleNamespace(
                config=SimpleNamespace(session_id="unfinished"),
                state="incomplete",
                threads=[SimpleNamespace(is_alive=lambda: True)],
            ),
        )
        job = client.post(
            "/sessions/unfinished/remove", json={"confirm_session_id": "unfinished"}
        ).json()
        for _ in range(100):
            job = client.get("/jobs/" + job["id"]).json()
            if job["status"] == "failed":
                break
            time.sleep(0.02)
        assert job["status"] == "failed" and root.exists()
        assert "active acquisition threads" in job["error"]


def test_failed_disk_cleanup_retains_recoverable_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATA", tmp_path)
    monkeypatch.setattr(main, "recorder", None)
    root = tmp_path / "sessions/cleanup_failure"
    root.mkdir(parents=True)
    atomic_json(
        root / "metadata.json",
        {
            "schema_version": "1.0",
            "session_id": "cleanup_failure",
            "status": "complete",
        },
    )
    with TestClient(main.app) as client:

        def disk_failure(path):
            raise OSError("Simulated disk cleanup failure")

        monkeypatch.setattr(main.shutil, "rmtree", disk_failure)
        job = client.post(
            "/sessions/cleanup_failure/remove",
            json={"confirm_session_id": "cleanup_failure"},
        ).json()
        for _ in range(100):
            job = client.get("/jobs/" + job["id"]).json()
            if job["status"] == "failed":
                break
            time.sleep(0.02)
        assert job["status"] == "failed"
        assert not root.exists()
        assert (
            len(list((tmp_path / "app/removing_sessions").glob("*/metadata.json"))) == 1
        )
        assert pq.read_table(tmp_path / "manifest.parquet").num_rows == 0
