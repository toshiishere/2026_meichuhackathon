"""A recorded video a browser can actually play, without touching raw data.

The collector writes fragmented MP4 (`frag_keyframe+empty_moov`) so a killed
recording still holds every finished fragment. Recordings made before the
collector also wrote a `sidx` carry no index at all, and a browser then has to
read the entire file before it can report a duration or seek: on a 17-minute
session that is minutes of blank player. Indexing a copy once, into `derived/`,
makes replay instant while `raw/` stays exactly as recorded.
"""

import json
import os
import struct
from pathlib import Path

from apps.common.storage import atomic_json, utc_now

SOURCE = "raw/video.mp4"
INDEXED = "derived/video.mp4"
RECORD = "derived/video.json"


def top_level_boxes(handle, size):
    offset = 0
    while offset + 8 <= size:
        handle.seek(offset)
        header = handle.read(8)
        if len(header) < 8:
            return
        length, kind = struct.unpack(">I4s", header)
        if length == 1:
            length = struct.unpack(">Q", handle.read(8))[0]
        if length < 8:
            return
        yield kind.decode("latin1"), offset, length
        offset += length


def needs_index(path):
    """True for a fragmented recording with no segment index to seek by."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        fragmented = False
        for kind, offset, length in top_level_boxes(handle, size):
            if kind == "moov":
                handle.seek(offset)
                # `mvex` marks a fragmented file: the samples live in `moof`
                # boxes, not in this header's tables.
                fragmented = b"mvex" in handle.read(min(length, 1 << 20))
            elif kind == "sidx":
                return False
            elif kind == "moof":
                break  # A global index is written ahead of the first fragment.
    return fragmented


def source_identity(path):
    status = path.stat()
    return dict(source=SOURCE, bytes=status.st_size, mtime_ns=status.st_mtime_ns)


def ensure_playable_video(session):
    """Return the session-relative video a browser can seek, indexing if needed.

    Raw recordings are never modified: an unseekable one is stream-copied (no
    re-encode) into derived/video.mp4 with a front index, and reused afterwards.
    """
    source = session / SOURCE
    if not source.is_file() or source.is_symlink():
        raise ValueError("Session has no raw/video.mp4 to play")
    if not needs_index(source):
        return SOURCE
    indexed, record = session / INDEXED, session / RECORD
    identity = source_identity(source)
    if indexed.is_file() and not indexed.is_symlink() and record.is_file():
        try:
            if json.loads(record.read_text()).get("identity") == identity:
                return INDEXED
        except ValueError:
            pass

    import av

    indexed.parent.mkdir(parents=True, exist_ok=True)
    temporary = indexed.with_name("." + indexed.name + ".tmp")
    try:
        with av.open(str(source)) as inp:
            # The temporary name carries no extension the muxer could infer.
            with av.open(
                str(temporary), "w", format="mp4", options={"movflags": "faststart"}
            ) as out:
                streams = {
                    s.index: out.add_stream_from_template(s) for s in inp.streams
                }
                for packet in inp.demux():
                    if packet.dts is None:
                        continue  # Flush packet at end of stream.
                    packet.stream = streams[packet.stream.index]
                    out.mux(packet)
        os.replace(temporary, indexed)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    atomic_json(
        record, dict(identity=identity, built_at=utc_now(), movflags="faststart")
    )
    return INDEXED
