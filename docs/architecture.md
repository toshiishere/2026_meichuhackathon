# Collection architecture

The collection service records synchronized raw CSI and camera data. The optional
training service labels completed videos and fine-tunes the supplied CSI model;
model processing never runs in acquisition threads.

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

Phone frame index schema 1.2 preserves the raw phone clock, actual host receipt,
and estimated host capture times separately. Resumable clients calibrate a fixed
midpoint clock offset before capture and retain it across reconnects. Camera
exposure latency, asymmetric transport delay and clock drift remain limitations.
The video timeline and training alignment use capture time, so replay does not
compress a network outage into a burst of video.

Phone capture is paced by new video frames, independently of acknowledgements.
One JPEG encode and at most four outstanding frames at 15 FPS (eight at 30 FPS)
bound active work. Unacknowledged JPEGs are persisted in a 512 MiB IndexedDB buffer;
network stalls trigger reconnect and ordered, idempotent replay. Recorder queues
apply backpressure; preview queues can drop frames. Session stop waits up to 600
seconds for captured phone frames to flush. OffscreenCanvas JPEG encoding avoids
idle-task scheduling, with a synchronous older-browser fallback. WebSocket JPEG
compression is disabled in both service images.

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


The Train workspace uses a separate internal FastAPI service and ROCm container.
The backend proxies an explicit training API allowlist. The worker holds a process
lease on the GPU job database and a per-session file lock shared with session
removal. It launches one cancellable subprocess per job, persists status in
`app/training.sqlite`, and captures stage progress/logs under `train/runs/<job-id>`.
No Docker socket or host virtual environment is exposed. Training endpoints remain
off the public phone listener.

Pose labeling decodes actual MP4 PTS and verifies every frame against the saved
Parquet index. Label times are mapped to schema 1.2 capture timestamps (falling back
to legacy host receipt timestamps). CSI preprocessing rejects unsupported layouts
and poorly covered windows; it never modifies raw files. The supplied ResNet18
backbone loads from a session-local copy of the original checkpoint. Its classifier
is replaced with a head for this session's labels. All receivers and overlapping
windows in a labeled interval remain together in the train/validation split.
Complete stage outputs are published separately; immutable run directories and an
atomic `train/model.json` pointer identify the matching model, classes, labels,
settings and metrics. Interrupted jobs retain their partial run and become failed
on worker restart. See README for output paths and deployment commands.

## Deployment and replay

The ROCm worker also owns a single deployment controller, sharing its operation
mutex with training. It uses ROCm directly or sends normalized windows to the
isolated Ryzen AI service for XINT8 inference. Deployment holds the selected
model and replay-session file locks
until completion. It pins the immutable run's checkpoint/classes and verifies the
checkpoint hash before loading the complete fine-tuned state, including its head.
The NPU service runs from the licensed Ryzen AI virtual environment in a separate
Ubuntu 24.04 container with exclusive `/dev/accel/accel0` access. On first use it
exports ONNX and calibrates Quark XINT8 from class-balanced training windows. The
FP32/XINT8 models, provenance and Vitis AI compilation cache live beside the
immutable run checkpoint and are reused after checksum validation.

For live input the worker requests an exclusive capture lease from the hardware
service. That service uses the existing binary/CSV serial framer, reads each
selected receiver on its own thread into its own bounded queue, stamps packets on
receipt, and exposes per-receiver batches over the internal API. One receiver
failing is reported per receiver; the lease ends only when every receiver has
stopped. Camera acquisition runs separately under the same lease, at the camera's
own frame rate, keeping a few seconds of timestamped JPEGs so a frame can be
served for a requested CSI time. A 15-second poll timeout releases hardware if
the worker disappears. No model code runs on serial acquisition threads and no
serial device is exposed to the ROCm container.

Inference uses shared training packet validation and timestamp resampling, then
the training loader's normalization. Each receiver has its own bounded rolling
window; every live receiver is resampled onto the same window end. ROCm scores
all receivers in one batch, while the fixed-batch NPU service scores them
sequentially. Their class probabilities are averaged into a single pose (score
fusion of the shared single-link backbone that training used
on every receiver's windows). Unsupported layouts and inadequate coverage do not
produce predictions for that receiver, and the pose is fused from the rest.
Recorded replay streams collector CSV/zstd rows for every selected receiver on
one shared original host timeline at the selected speed, through the same
rolling-window path. Before publishing the replay video the worker makes sure it
is seekable: a fragmented recording without a segment index is stream-copied
once into the session's `derived/video.mp4` (raw data untouched) and the browser
is pointed at whichever file carries an index. Labels/video are not used as inference inputs. The UI polls
deployment status and camera images separately; a lost UI connection hides its
current prediction but does not stop deployment.

Fall alerting runs in the worker, not the browser, so a replay alerts with no UI
attached. It watches the fused predictions on the source clock and posts to the
`dc-bot` service, which owns the Discord gateway session and the bot token; the
worker holds no credentials, and a send failure is recorded in deployment status
without touching the inference loop.
