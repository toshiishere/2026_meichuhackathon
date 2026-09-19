"""Finalize closed, incomplete recordings without rewriting their raw content."""

import csv
import io
import json
import re

import av
import pyarrow.parquet as pq
import zstandard as zstd

from apps.common.schemas import ID
from apps.common.storage import atomic_json, rebuild_manifest, utc_now


def recover_session(root, sid, log, stop):
    path = root / "sessions" / sid
    raw = path / "raw"
    if raw.is_symlink() or not raw.is_dir() or (path / "metadata.json").is_symlink():
        raise ValueError("Unsafe or missing recording directory")
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("status") != "incomplete":
        raise ValueError("Only incomplete sessions can be recovered")
    receivers = metadata.get("configuration", {}).get("receivers", [])
    names = [r["logical_name"] for r in receivers]
    if not names or any(not re.fullmatch(ID, name) for name in names):
        raise ValueError("Recording has no valid receiver configuration")
    expected = ["video.mp4", "video_frames.parquet"] + [
        f"csi_{name}.csv.zst" for name in names
    ]
    files, renames = {}, []
    for name in expected:
        final = raw / name
        candidates = [final, raw / f".{name}.tmp", raw / f"{name}.tmp"]
        if any(p.is_symlink() for p in candidates):
            raise ValueError(f"Unsafe artifact: {name}")
        existing = [p for p in candidates if p.exists()]
        if len(existing) != 1:
            raise ValueError(
                f"Missing or conflicting artifacts for {name}; nothing was renamed"
            )
        source = existing[0]
        if not source.is_file() or not source.stat().st_size:
            raise ValueError(f"Empty or invalid artifact: {source.name}")
        files[name] = source
        if source != final:
            renames.append((source, final))

    # A rename cannot repair unclosed containers. Validate before changing any file.
    frames = pq.read_table(files["video_frames.parquet"]).to_pydict()
    pts = frames["video_pts_s"]
    stamps = frames.get("capture_timestamp_ns", frames.get("host_timestamp_ns", []))
    if (
        not pts
        or len(stamps) != len(pts)
        or any(s is None for s in stamps)
        or any(b <= a for a, b in zip(pts, pts[1:]))
        or any(b <= a for a, b in zip(stamps, stamps[1:]))
        or frames["frame_idx"] != list(range(len(pts)))
    ):
        raise ValueError(
            "Frame index is incomplete or unordered; rename alone cannot recover it"
        )
    count, previous = 0, None
    with av.open(str(files["video.mp4"])) as video:
        for packet in video.demux(video.streams.video[0]):
            if stop.is_set():
                raise RuntimeError("Recovery cancelled; nothing was renamed")
            if packet.pts is not None:
                if previous is not None and packet.pts <= previous:
                    raise ValueError("Video timestamps are unordered")
                previous = packet.pts
                count += 1
    if count != len(pts):
        raise ValueError("Video/frame index mismatch; rename alone cannot recover it")
    for name in expected[2:]:
        count = 0
        with files[name].open("rb") as source:
            with zstd.ZstdDecompressor().stream_reader(source) as reader:
                with io.TextIOWrapper(reader) as text:
                    for row in csv.DictReader(text):
                        if stop.is_set():
                            raise RuntimeError(
                                "Recovery cancelled; nothing was renamed"
                            )
                        int(row["host_timestamp_ns"])
                        if not row.get("data"):
                            raise ValueError(f"Missing CSI payload in {name}")
                        count += 1
        if not count:
            raise ValueError(f"No CSI packets in {name}")
    if stop.is_set():
        raise RuntimeError("Recovery cancelled; nothing was renamed")
    for source, final in renames:
        source.rename(final)
        log(f"Recovered raw/{source.name} → raw/{final.name}")
    # Keep the original errors/statistics visible; recovery is not a quality guarantee.
    metadata.update(
        status="complete",
        quality="recovered",
        recovered=True,
        recovered_at=utc_now(),
        recovery=dict(
            previous_status="incomplete",
            renamed=[str(p.relative_to(path)) for _, p in renames],
        ),
        files=[
            dict(path=str(p.relative_to(path)), bytes=p.stat().st_size)
            for p in raw.iterdir()
            if p.is_file()
        ],
    )
    atomic_json(path / "metadata.json", metadata)
    rebuild_manifest(root)
    return dict(session_id=sid, recovered=True, renamed=len(renames))
