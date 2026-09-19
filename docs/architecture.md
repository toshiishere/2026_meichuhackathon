# Collection architecture

Scope: prompt milestones 1–6 only. The present deliverable is synchronized raw
CSI and camera collection and inspection. No labeling, learning, inference,
window datasets, or downstream feature processing is implemented.

Browser (React/TypeScript) → nginx → FastAPI backend → internal hardware API.
Only the hardware service can access devices. Backend stores logical assignments
and hardware service stores jobs in separate SQLite databases. Session directories
are canonical; the session index and root Parquet manifest are rebuildable.

The collector uses one serial acquisition thread and bounded queue per receiver,
one camera acquisition thread and bounded queue, separate streaming file writers,
and one shared monotonic start gate. Readers timestamp at complete serial-line
receipt / successful camera read, before queueing. Binary CSI is timestamped at
receipt of the host serial chunk completing the frame; firmware acquisition time
is retained separately. The host clock synchronizes
modalities; transmitter sequence numbers synchronize receivers. USB buffering and
camera driver latency remain measurement limitations, not corrected estimates.

Shutdown stops readers, drains queues, closes all writers, checks output counts,
and atomically renames completed files. A failed writer or empty stream produces
an incomplete session. Overflows are counted and mark collection degraded.
Startup finds interrupted metadata and marks it incomplete without deleting data.
CSI compressed streams periodically flush independent frames for crash inspection;
fragmented MP4 preserves already flushed fragments. Temporary artifacts are kept.
Raw files are never overwritten by a new session; names are reserved using mkdir.

Hardware access has a service-wide lease: tests, probe, builds/flashes, and recording
cannot overlap. Camera preview owns the lease until its stream closes. Recording
runs outside HTTP request handling. The backend provides SSE status and job updates
once per second; it does not handle acquisition packets.

Firmware was inspected before implementation: existing local source has upstream
commit 8633d67152db2808f141cc1595970aa9cf406045, generated SDK configs identify IDF
5.5.0, sender target esp32, receiver esp32c3. Sender and receiver app_main.c match
upstream examples. Vendored project copies retain SDK configs, dependency locks,
100 Hz, channel 11, HT20 and RX 921600 baud. Original tree and boards are untouched.
Firmware cache keys include source content, original upstream commit, target,
IDF version and blink configuration. Builds use private copies of the projects.

Phone cameras connect through an optional nginx TLS listener, which serves only
`/phone`, static assets, the lab CA certificate, and `/api/phone/stream`. Phone
pairing is created through the local administration API. The upload WebSocket goes
directly to the hardware service; the backend does not forward acquisition frames.
The hardware service checks one-use expiring credentials, timestamps each complete
message before JPEG decoding, acknowledges each accepted frame, and delivers decoded
frames through bounded per-camera subscriber queues. The regular camera interface
supports preview, preflight, recording, and disconnect/stall failure handling.

Phone sequence and `performance.now()` capture times are retained as nullable frame
index schema 1.1 columns. They never replace the shared host clock. Native phone
resolution is shown in discovery; the browser scales with letterboxing to the paired
output size. Sensor-to-host latency remains unmeasured; the phone displays local
encoding duration and send-to-acknowledgement delay for diagnosis. Client skips and queue drops are
reported. No clock-offset correction or synthetic frame interpolation is applied.

Phone capture is paced by new video frames, independently of acknowledgements.
One JPEG encode and at most four unacknowledged frames at 15 FPS (eight at 30 FPS)
bound the browser work. Frames are skipped if encoding or transmission capacity is
full. A five-second acknowledgement/encoding watchdog stops stalled uploads.
OffscreenCanvas JPEG encoding avoids idle-task scheduling, with a synchronous
older-browser fallback. WebSocket JPEG compression is disabled in both service images.

Serial discovery filters actual tty nodes to `/dev/ttyUSB[0-9]` and
`/dev/ttyACM[0-9]`, retaining available stable by-id aliases. Validation checks the
current discovered set before hardware operations.

Explicit session removal takes the exclusive hardware lease, rejects active
recordings, atomically moves the directory out of the archive into
`app/removing_sessions`, then deletes it and rebuilds the manifest. If cleanup
fails, remaining files stay there for manual recovery and the job records failure.
The backend prunes removed session index entries when the archive is refreshed.

The 2026-09-13 firmware patches add a fixed-pool CSI callback queue, a lower-priority
binary output task, direct bulk USB/UART driver output, and a periodic sender schedule. The
collector accepts old CSV and new CRC-protected binary frames in the same framing
layer. Raw binary frames are retained alongside decoded metadata and original int8
samples. See [wire format and loss accounting](csi-binary-v1.md).

Sender selection is optional: battery-powered transmitters do not need USB.
Preflight validates any explicitly selected sender's USB identity; all modes
require actual CSI arrival from each receiver. Receiver sequence alignment and
host monotonic timestamps are unchanged. A session stores a null sender when
external power is selected. Device registrations can be removed independently
of historical session configuration snapshots. The web list joins registrations
to current serial discovery by identity and labels stale port information.

USB capture disables exposure-driven dynamic frame rates by default, keeping auto
exposure enabled. The option applies to preview, tests, and recording. Negotiated
FPS, dimensions, pixel format, and exposure controls are included in diagnostics;
metadata-only UVC device nodes are omitted from camera discovery.
