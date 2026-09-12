import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from .config import SCHEMA_VERSION


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value):
    temp = path.with_name("." + path.name + ".tmp")
    with temp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def read_session(root: Path, session_id: str):
    path = root / "sessions" / session_id / "metadata.json"
    value = json.loads(path.read_text())
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported session schema version")
    return value


def scan_sessions(root: Path):
    values = []
    for p in sorted((root / "sessions").glob("*/metadata.json"), reverse=True):
        try:
            value = read_session(root, p.parent.name)
        except (ValueError, OSError) as e:
            value = {
                "session_id": p.parent.name,
                "status": "unreadable",
                "error": str(e),
            }
        values.append(value)
    return values


MANIFEST_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("session_id", pa.string()),
        ("subject_id", pa.string()),
        ("room_id", pa.string()),
        ("layout_id", pa.string()),
        ("date", pa.string()),
        ("duration", pa.float64()),
        ("receiver_count", pa.int64()),
        ("camera", pa.string()),
        ("status", pa.string()),
        ("quality", pa.string()),
    ]
)
manifest_lock = threading.Lock()


def rebuild_manifest(root: Path):
    with manifest_lock:
        root.mkdir(parents=True, exist_ok=True)
        rows = []
        for s in scan_sessions(root):
            c = s.get("configuration", {})
            rows.append(
                dict(
                    schema_version=SCHEMA_VERSION,
                    session_id=s["session_id"],
                    subject_id=c.get("subject_id", ""),
                    room_id=c.get("room_id", ""),
                    layout_id=c.get("layout_id", ""),
                    date=s.get("created_at", ""),
                    duration=s.get("duration_seconds", 0),
                    receiver_count=len(c.get("receivers", [])),
                    camera=c.get("camera", {}).get("device", ""),
                    status=s["status"],
                    quality=s.get("quality", "unknown"),
                )
            )
        temp = root / ".manifest.parquet.tmp"
        pq.write_table(pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA), temp)
        os.replace(temp, root / "manifest.parquet")
    return {"sessions": len(rows)}


class Registry:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS records (kind TEXT, id TEXT, value TEXT, PRIMARY KEY(kind,id))"
            )

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def put(self, kind, key, value):
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO records VALUES (?,?,?)",
                (kind, key, json.dumps(value)),
            )

    def get(self, kind, key):
        with self.connect() as db:
            row = db.execute(
                "SELECT value FROM records WHERE kind=? AND id=?", (kind, key)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, kind):
        with self.connect() as db:
            rows = db.execute(
                "SELECT value FROM records WHERE kind=? ORDER BY rowid DESC", (kind,)
            ).fetchall()
        return [json.loads(row[0]) for row in rows]
