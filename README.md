# CSI Collection Lab 

A Docker-first web application for **synchronized WiFi CSI and USB or phone camera collection**.
Configure ESP32 boards, verify receiver/camera health, record multiple receivers
and a camera together, and inspect immutable raw sessions.

Collection remains independent of model processing. The **Train** workspace
adds session video labeling and CSI fine-tuning in a separate ROCm container.
**Deploy** uses that worker for live CSI inference and recorded-session replay,
and `make up` starts the worker with the collection services. `make mock` starts
collection services without the GPU worker.

## Quick start — without hardware

Install Docker Engine and Docker Compose **2.24.4+** on Linux. Nothing else is
required on the host. The mock override uses Compose's `!override` syntax.

```bash
cp .env.example .env
make mock
```

Open **http://localhost:8080**. If your account needs sudo for Docker:

```bash
make mock DOCKER='sudo docker'
```

In **Hardware Setup**:

1. Optionally select `synthetic://tx`, assign `tx_main`, role **CSI Sender**, and save.
2. Select `synthetic://rx0`, assign `rx_left`, role **CSI Receiver**, and save.
3. Optionally assign `synthetic://rx1` as `rx_right`.
4. Run **CSI test / monitor** on a receiver. Open the job result and logs.
5. Open the camera preview, then close it and run **Test camera**.
6. In **Data Collection**, select the receivers and camera. Leave the optional sender
   at **External / battery-powered sender (no USB)** unless you want to track a USB-connected sender. Enter a unique
   session ID and experiment details. Run preflight, then **Start recording**.
7. **Stop recording** whenever desired, or set an optional duration. Wait for
   `complete`, then open **Sessions** for playback, metadata, quality, and downloads.

Synthetic recording defaults to 100 CSI packets/sec/receiver and 1% seeded packet
loss. Loss and seed are configurable. Synthetic cameras render a moving marker,
frame number and monotonic time. Sessions explicitly record `hardware_mode`.
Synthetic mode never reports a physical probe or firmware flash as successful;
flashing is unavailable there.

## Labeling and fine-tuning

The **Train** workspace operates on a completed session:

1. Select a session and click **Labeling**. `pose_labeling/inference.py` runs
   YOLOv8-Pose and the supplied four-class CTR-GCN checkpoint on the session's
   `raw/video.mp4`, using the AMD GPU. The existing motion gate can also emit
   `Static`. Record one clearly visible person.
2. Download/review `train/action_results.csv`. `start_time` and `end_time` are
   **milliseconds on the video's PTS timeline**, not frame count divided by FPS.
   `start_host_timestamp_ns` and `end_host_timestamp_ns` map these boundaries to
   the collector capture clock using `raw/video_frames.parquet`. Schema 1.2's
   `capture_timestamp_ns` handles replayed phone frames; older indexes use
   `host_timestamp_ns`. Missing poses and camera gaps break the labeling history.
3. Click **Fine-tune** to preprocess `raw/csi_*.csv.zst` (or `.csv`) and initialize
   a session copy of `csi_model/Model Code/pretrained/ResNet18_full.pth`. The supplied
   `ESP_Fi_ResNet18` backbone is retained; its classifier is replaced for the actions
   present in this session. All supported actions are included, not just three.
4. **Auto train** performs labeling → preprocessing → fine-tuning in one job.
   Watch stage/log output or cancel the job. A failed stage stops the sequence.

The default is two-second windows, 100 Hz resampling, 50% overlap, batch size 8,
five epochs training the new classifier and fifteen epochs fine-tuning the whole
model. The UI exposes window size, batch size and epoch counts. Each receiver
supplies single-link examples to the same pretrained model; this is not a new
multi-input architecture. Preprocessing extracts 52 L-LTF amplitude subcarriers
and the loader applies the existing per-window global z-score normalization.
Raw recordings are read directly without decompressing additional files into `raw/`.
Only the supplied decoder's non-STBC HT20/256-byte and HT40/384-byte layouts with a
valid first word are accepted. Rejected packets, gaps and short intervals are
reported in preprocessing metadata. Standalone wire binary without recorded host
receipt timestamps cannot be aligned safely and is rejected.

At least two actions with usable windows are needed. Windows stay wholly inside
one labeled interval; intervals shorter than the selected window are omitted.
Camera gaps over 500 ms, CSI gaps over 200 ms, and windows below 70% packet coverage
are omitted. All receivers/windows from an interval stay in the same training or
validation split. If there are no independent validation intervals, the model
still trains and the UI explicitly reports **no validation score**. These are
machine-generated labels and same-session validation, not reviewed ground truth or
an estimate of performance on new people/rooms.

```text
data/sessions/session_XXX/train/
  action_results.csv            # latest completed labeling output
  action_results.json           # labeling/timestamp provenance
  pretrained_resnet18.pth        # copied baseline for latest successful fine-tune
  finetuned_resnet18.pth         # latest successful model state_dict
  classes.json                  # class names and output indices
  metrics.json                  # epoch losses and available validation metrics
  model.json                    # authoritative pointer/provenance for model run
  runs/<job-id>/
    options.json, job.log, progress.json
    action_results.csv          # when this run performed labeling
    labels_used.csv             # exact label snapshot used for training
    pretrained_resnet18.pth, finetuned_resnet18.pth, classes.json, metrics.json
    model.json, train_split.csv, validation_split.csv
    dataset/manifest.csv, dataset/preprocessing.json, dataset/<action>/*.mat
```

Every run starts from the original baseline. Previous run directories remain
available on failure/cancellation or reruns. `model.json` identifies the matching
run-specific checkpoint, class mapping and preprocessing settings; use that pair
for later inference. Session removal is blocked while its training job holds a
shared-storage lock. Worker restart marks unfinished jobs failed; it does not
silently resume an optimizer or overwrite a prior successful run.

### ROCm container and deployment

All training dependencies are installed by `docker/training.Dockerfile`; neither
host `.venv` is used or copied. It pins AMD's ROCm 10 wheels for **gfx1152**, matching
this machine's Radeon 860M, and the provided CTR-GCN source revision. Both pose
labeling and CSI training fail clearly if a ROCm GPU is unavailable; there is no
CPU fallback. The image includes the C library headers MIOpen needs for runtime
kernel compilation. See the [AMD installation guidance](https://rocm.docs.amd.com/en/docs-10.0.0/install/rocm.html)
and [Ultralytics ROCm integration](https://docs.ultralytics.com/integrations/amd).

Build only, without deploying:

```bash
docker compose -f docker-compose.yml -f docker-compose.train.yml build training-service
```

`make up` builds and starts all collection services and the ROCm training worker.
Use `sudo make up` if Docker requires root. `make down` and `make logs` also include
the training worker. To deploy with the existing phone portal enabled:

```bash
make up
make phone
```

Omit `make phone` if the phone portal is not in use. `make up` rebuilds the backend,
frontend and hardware service as well as the worker so the Train API/UI and
session-removal lock are installed together. The training worker has no public port, no Docker
socket, and accesses the GPU through `/dev/kfd` and `/dev/dri`. One job uses the GPU
at a time. Set `ROCM_GPU` in `.env` before building for a different supported GPU;
the host AMD driver must support it. The source tree must contain the supplied
`pose_labeling/yolov8n-pose.pt`, `pose_labeling/ctrgcn_custom_4classes_best.pth`, and
CSI pretrained checkpoint. No models are downloaded by a recording/training job.

For container-only command-line labeling after building:

```bash
docker compose -f docker-compose.yml -f docker-compose.train.yml run --rm --no-deps training-service \
  python pose_labeling/inference.py --session /data/sessions/session_XXX
```

The web worker adds concurrency protection and run history; use the direct CLI
only when no job is using that session.

## Deploy and replay a trained model

Rebuild the stack with `sudo make up` after updating. If using the phone portal,
run `sudo make phone` afterward. Deployment uses the existing ROCm worker and
does not install dependencies into a host venv.

1. Open **Deploy** and choose a **Model session** with a completed fine-tune.
   The worker loads the immutable checkpoint and matching class mapping referenced
   by `train/model.json`, verifies the checkpoint hash, and runs the model on ROCm.
2. Choose **Live serial receiver** and a connected, registered receiver. The
   existing serial framer reads CRC-checked binary CSI (legacy CSV also works).
   Keep the ESP32 sender powered; it does not need a USB connection to this computer.
3. Optionally select a **Live camera**. USB camera settings follow Hardware Setup;
   paired phones retain the resolution/FPS of their phone link. The preview is
   independent of model input. A camera failure is reported without stopping CSI
   inference.
4. Click **Start deployment**. The page shows the predicted action, all class
   scores, packet diagnostics and recent predictions. These are the model's action
   classes, not pose keypoint coordinates. Scores are model probabilities, not
   calibrated accuracy measurements.
5. Click **Stop deployment** to release the receiver, camera, GPU and session
   locks. Leaving the page keeps deployment running; return to Deploy to stop it.

For a demo, choose **Recorded session replay**, then select the same or another
completed session and one of its recorded receivers. Replay streams the original
`raw/csi_*.csv.zst` or `.csv` at 0.25×–4× speed using recorded host timestamps,
including gaps, without loading the whole session into memory. It runs the actual
fine-tuned model, does not substitute video labels, and stops at the end of the
recording. A live camera, if selected during replay, shows the present scene and
is not synchronized to the historical recording. Replaying training data is a
demo, not an independent accuracy evaluation.

Training and deployment share the same packet decoder, 52 L-LTF amplitude
selection, timestamp resampling and per-window global z-score normalization.
Deployment takes the window duration, sample rate and update stride from the
trained model's metadata, producing `[1, 1, time_samples, 52]` tensors. It requires
70% packet coverage, bracketing timestamps and no CSI gaps above 200 ms. It waits
for a complete window before predicting and clears the current prediction during
missing data. At present each prediction uses one receiver, matching the trained
single-link backbone. Camera frames never enter this model.

Training and deployment cannot occupy the GPU simultaneously. Live deployment
owns the hardware lease, preventing recording, flashing and competing previews.
Model/replay sessions cannot be deleted or retrained while in use. If the worker
disconnects, the collector stops deployment acquisition after 15 seconds without
polling. Deployment does not modify raw sessions or save new recordings.

## Real hardware

Required: an ESP32 sender and one or more ESP32 receivers running the preserved
Espressif CSI projects, USB serial connections for the receivers, and either a Linux USB/UVC camera or a phone browser camera.
The sender can run from a battery bank with its USB disconnected from the collector.
Flash it first, then power it on before preflight; all boards must use the same WiFi channel.

```bash
make down
make up
# or, without Make:
docker compose -f docker-compose.yml -f docker-compose.train.yml up -d --build
```

The normal Compose file uses **`espressif/idf:v5.5`**, matching the ESP-IDF 5.5.0
headers and dependency locks in your existing working projects. The first real
hardware image build downloads the ESP-IDF toolchains and can take several minutes.
See [firmware provenance](firmware/README.md) and
[Espressif's official Docker documentation](https://docs.espressif.com/projects/esp-idf/en/v5.5/esp32/api-guides/tools/idf-docker-image.html).

The working source was found in `/home/toshi/esp-csi` and copied into this repo;
no host path dependency remains. The original repository and boards were not
modified. Receiver output: **921600 baud**. Sender: **100 Hz**, **channel 11**,
**HT20**, source MAC `1a:00:00:00:00:00`. See [firmware/README.md](firmware/README.md)
for the preserved commit, SDK configs, targets and hashes.

### Identify and flash boards

Refresh Hardware Setup after reconnecting USB devices. Discovery only includes
`/dev/ttyUSB0`–`/dev/ttyUSB9` and `/dev/ttyACM0`–`/dev/ttyACM9`; `/dev/ttyS*`
is excluded. This filters the application list; it does not remove Linux device nodes. Stable `/dev/serial/by-id`
paths and USB identities preserve assignments across tty renumbering. Devices
without unique USB serials fall back to stable paths, USB location, then tty path;
verify these assignments again after moving USB sockets.

**Registered logical names** lists every saved assignment, including disconnected
boards, its role, target, USB identity, and current port. Disconnected entries show
the last registered port separately. Click **↻ Refresh** after reconnecting.
**Remove registration** frees a name for reuse without deleting saved sessions or
changing board firmware. Reconnected boards whose registration was removed appear
as unassigned.

**Probe board** runs esptool and returns actual chip/MAC/flash output. It does not
write flash, but entering the bootloader may reset the board. Choose its actual
chip target before building or flashing. Supported selectable targets are ESP32,
ESP32-C3, ESP32-S3 and ESP32-C6; only the original ESP32 sender and ESP32-C3 receiver
configurations were found locally.

For visual identification choose **Blink / Identify**, choose GPIO or WS2812 RGB,
and supply a valid LED GPIO from the board pinout. Active-low GPIO LEDs are
supported. A board without a usable LED cannot be identified by blinking. Select
**Replace firmware on the selected board**, then flash. Observe the physical LED,
save its name/role, and restore **CSI Sender** or **CSI Receiver** firmware.
A completed flash confirms the command succeeded; visually verify the LED and
run a CSI test to verify application behavior.

Builds run from isolated cached project copies. `Build` reuses a successful cache;
`Rebuild` and `Clean build` rebuild it; `Flash firmware` builds only if needed.
The cache incorporates source contents, upstream commit, IDF version, target, and
LED configuration. No shell strings are accepted from the browser.

Firmware operations, serial tests, camera tests/previews, and recording share an
exclusive hardware lease. Close camera preview before testing or recording.
Probe/flash/test logs appear in the UI and under `data/app/job_logs`.

### Linux device permissions

The initial hardware service is privileged and mounts `/dev`. **This gives that
container broad access to host devices.** It is intended for a trusted local lab.
The backend and frontend have no device mounts. Only nginx is exposed, bound to
`127.0.0.1:8080`; do not expose this unauthenticated application publicly.

A hardened installation can remove `privileged` and `/dev`, map only the required
serial and camera devices with Compose `devices`, and add the host `dialout` and
`video` group IDs. Hotplug and stable-path symlinks must also be accessible inside
the container. `make hardware-shell` opens the real hardware environment.

If serial/camera access fails, close other serial monitors or camera apps,
check cable/data support and `/dev` visibility, and inspect hardware job logs.
No operation silently substitutes synthetic input for failing physical hardware.

## Use a phone as the recording camera

No domain is required. In `.env`, set `PHONE_HOST` to an IPv4 address or hostname
that the phone can reach and resolve, without `https://` or a port:

```dotenv
PHONE_HOST=192.168.1.50
PHONE_HTTPS_PORT=8443
PHONE_CERT_PORT=8081
```

A reachable public IPv4 address also works. Routing/firewall access to these ports
must be available from the phone. A hostname must already resolve on that network;
this setting does not register a domain or configure DNS.

Start the normal or mock stack first, then:

```bash
make phone-cert           # build frontend and generate certificates in Docker
make phone                # enable phone ports; preserve current hardware mode
# Add DOCKER='sudo docker' if needed.
```

1. On the phone, download `http://YOUR_PHONE_HOST:8081/phone-ca.crt`. Install the
   lab CA certificate as trusted, using the phone's certificate settings. On iOS,
   also enable full trust under **Settings → General → About → Certificate Trust
   Settings**. The certificate must be trusted by the browser. A warning bypass
   alone may not enable camera capture; [browser camera access requires a secure
   context](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia).
2. On the computer at `http://localhost:8080`, open **Hardware Setup → Connect a
   phone camera**. The address comes from `.env`; choose a name and capture preset,
   then **Create phone pairing link**. Scan the displayed QR code with the phone
   or copy the link. The QR code is generated locally, including the one-use token.
3. On the phone, tap **Choose camera / preview**, allow camera access, and select
   a lens from **Camera**. Automatic selection prefers an identified rear ultra-wide
   lens, then wide-angle, and falls back to the default rear camera. Available lenses
   depend on what the browser exposes; [camera enumeration requires permission](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/enumerateDevices).
   Previewing and switching lenses do not consume the pairing link. Choose the lens
   before streaming, then tap **Start phone camera**. You can also start directly
   with automatic selection.
   The live view and delivered-frame counter show the upload is running. No audio
   is requested or recorded.
4. On the computer, **Refresh connected cameras**, select the phone in **Capture
   device**, and run preflight/recording as usual. The paired resolution and FPS
   apply automatically. Close a computer preview before preflight or recording.
   Camera Setup shows the selected phone's paired width, height, and FPS, including
   when it is automatically selected on page load. These fields are locked because
   the phone sends frames using its pairing preset. To change them, create a new
   link with the desired preset, reconnect the phone, refresh cameras, and select
   the new connection. Editing the preset for a future link does not reconfigure
   an existing phone stream.
5. Keep the phone page visible and the phone awake throughout recording. Stop the
   recording on the computer before stopping the phone camera. During a network
   outage, capture continues into IndexedDB on the phone and the page reconnects
   automatically with the same camera identity. The saved-frame counter shows the
   upload backlog. CSI collection continues throughout the outage.
6. After stopping a session, the collector waits up to **600 seconds** for phone
   frames captured before the stop time (`phone_sync_timeout_seconds` in
   `configs/collection.yaml`). Keep the phone page open until the session finishes.
   **Stop phone camera** finishes capture and uploads its remaining saved frames.
   If the page reloads, use **Resume … saved frames** on the same phone and browser
   origin. Reloading itself interrupts capture; already saved frames survive.

Unacknowledged JPEGs are saved before upload and removed only after the collector
accepts them. Retries are ordered and duplicates are acknowledged without being
recorded twice. The local buffer is limited to **512 MiB**, or available browser
storage if lower. Capture pauses with a visible error if storage is full; existing
saved frames continue uploading. Browser storage eviction/clearing or losing the
phone cannot be recovered. Keeping the page open and screen awake is still required
for capture; frames never captured during browser suspension cannot be recreated.
Only one tab should own each phone stream (enforced through Web Locks where available).

Resume credentials last for 24 hours after the latest connection/heartbeat. They
are independent of the unused pairing link's 10-minute expiry. The collector must
remain running: restart loses its in-memory phone registration and ends an active
session. If final upload times out, the session is incomplete with partial artifacts;
late uploads cannot be added to that finalized session. Local unacknowledged frames
are retained, but recovery after a collector restart or timeout is not automatic.

The phone listener exposes only the phone page, certificate, and token-protected
camera upload. Pair creation, device operations, recording controls, and session
files remain on the localhost administration listener. Pairing links expire after
10 minutes and can be used once. Only share the link with the intended camera;
only `ca.crt` is for download, never the private keys in `PHONE_TLS_PATH`.

Phone frames are JPEG uploads, encoded to the same H.264 session video as USB
frames. Three initial clock probes select the lowest round-trip time; their midpoint
provides a fixed estimate mapping the phone capture clock to the CSI host clock.
Video PTS and session boundaries use that estimated capture time, so replaying a
backlog does not compress an outage into a burst of video. Raw phone capture times,
actual host receipt times, and the estimated offset/round-trip uncertainty are
retained separately. The estimate is not hardware synchronization: camera exposure
latency, scheduling delays, asymmetric networks and clock drift remain unmeasured.
Legacy phone clients retain receipt-time synchronization.

Capture and JPEG encoding run independently of acknowledgements, with at most four
outstanding frames at 15 FPS or eight at 30 FPS. A slow connection adds frames to
local storage instead of skipping capture. Recording queues apply backpressure to
replay; preview/test queues remain bounded and may drop frames. A busy JPEG encoder
can still skip capture intervals. Skips, queue drops and insufficient coverage/FPS
can mark a recording degraded. Actual FPS must pass preflight.

The phone page shows observed camera FPS, acknowledged delivery FPS, JPEG encoding
time, acknowledgement delay, and frames in flight. High acknowledgement delay
includes network travel and collector decoding; it is not a clock-offset estimate.
JPEGs use `OffscreenCanvas.convertToBlob` where available, with a synchronous JPEG
fallback for older browsers, avoiding idle-scheduled canvas encoding. WebSocket
compression is disabled because JPEG frames are already compressed. A slow phone
camera or insufficient upload bandwidth can still limit the achievable rate.

Change `PHONE_HOST` and rerun `make phone-cert` and `make phone` when the address
changes. The existing CA is reused. `make phone-off` removes the extra listeners.
After a later `make up` or `make mock`, rerun `make phone` to enable them again.

## Recording and synchronization

Preflight verifies all selected devices, identity consistency, actual parsable
CSI arrival and rate, requested camera resolution and measured FPS, writable
storage, and at least 1 GiB free disk (configurable). It runs again on every start.
No override bypasses failed checks. Sender selection is optional and defaults to
external power; its transmission is verified through received CSI packets. If a
USB-connected sender is explicitly selected, preflight also checks its identity
and presence. No sender serial stream is used for synchronization. External sender
sessions store `configuration.sender: null` and preflight reports `sender_connection: external`.

All devices and outputs initialize before a shared acquisition gate opens.
CSI packets and camera frames receive **`time.monotonic_ns()`** timestamps in the
same service process immediately after complete line receipt / camera read /
phone WebSocket frame receipt.
ESP local timestamps are microseconds (uint32, wrapping); UTC time is only descriptive. Serial queues, compression, video encoding, and HTTP
requests do not define serial or USB camera acquisition timestamps.

Use `tx_seq` to align receivers. Each raw row also preserves ESP local timestamp,
original `id`/`seq`, MAC, RSSI, all emitted PHY fields, gain context, and
`first_word`. Binary receiver firmware keeps original signed int8 samples and the
entire CRC-protected wire frame; legacy CSV keeps its gain-compensated int16 values
and original line. Each new row identifies its sample representation. Arrays are
not clipped, converted to amplitude, or assigned a fixed subcarrier count. Both
formats preserve **imaginary then real** ordering. See the [binary CSI transport
and firmware upgrade guide](docs/csi-binary-v1.md).

Sequence gaps, duplicates, backwards/reset events, parse errors, queue overflows,
per-receiver first/last timestamps and camera frame counts are recorded. A backwards
sequence can mean reset or reordering and is reported as ambiguous. No missing
samples are interpolated or removed from the record. Non-CSI lines and rejected
serial input are saved separately, including original bytes encoded as base64.

Video uses H.264 and variable presentation timestamps derived from actual host
frame acquisition times. Playback PTS zero is the first encoded frame; the frame
index includes both PTS/time base and its host clock timestamp. This preserves
camera jitter and gaps. Serial/USB buffering and camera driver latency remain
unmeasured; shared host timestamps do **not** imply hardware-trigger precision.

On stop, readers stop accepting input, writers drain, CSV streams flush, MP4 and
Parquet finalize, and video packet/frame-index counts are checked. Only then are
raw temporary files promoted and session metadata marked `complete`.

A completed session with packet loss, queue drops, parser failures, low rates, or
sequence resets is `degraded`. Hardware/encoder errors, no CSI or no video, and
finalization failures mark it `incomplete`. A data stall or low disk space stops
recording. No recording is overwritten or automatically deleted.

To explicitly delete a saved session, open **Sessions → Remove** and type its exact
session ID. This permanently removes its CSI, video, frame timestamps, metadata,
and collection logs, then refreshes the archive and manifest. Active recordings
and sessions with running acquisition threads cannot be removed. Removal is a
tracked job and cannot be cancelled after it starts.

## Data layout

Convert a saved CSI recording or raw binary capture to readable CSV and print
the first 10 lines (including the header):

```bash
python3 scripts/convert_csi.py path/to/csi_receiver.csv.zst -o readable.csv
python3 scripts/convert_csi.py path/to/capture.bin -o readable_binary.csv
```

The source stays unchanged. `.zst` input uses the `zstd` command or Python's
`zstandard` package; raw binary input uses the existing CRC-checked decoder.
See [conversion options](docs/csi-binary-v1.md#convert-an-existing-capture-to-readable-csv).

```text
data/
  manifest.parquet                 # rebuildable session index
  app/
    backend.sqlite                 # logical assignments, session index
    hardware.sqlite                # persistent jobs
    job_logs/
    firmware_cache/
  sessions/<session_id>/
    metadata.json                  # schema 1.0, config, provenance, stats
    raw/
      csi_rx_left.csv.zst           # streaming compressed raw CSV
      csi_rx_right.csv.zst
      video.mp4                    # fragmented H.264 MP4
      video_frames.parquet         # schema 1.2: frame index, PTS, capture/receipt/phone times
    logs/
      collection.log
      serial_rx_left.jsonl         # boot/gain lines and rejected records
      serial_rx_right.jsonl
      <failing-thread>.log         # traceback when an operation fails
```

New video frame indexes use schema `1.2`. `host_timestamp_ns` remains the actual
host receipt time; `capture_timestamp_ns` supplies the video timeline and session
boundary filtering. For resumable phones it equals `phone_estimated_host_capture_ns`,
with `phone_clock_uncertainty_ns` recording half the selected handshake round trip.
`phone_frame_idx`, `phone_capture_timestamp_ms`, and `phone_skipped_frames_total`
preserve phone provenance. For USB/legacy clients, capture time equals receipt time;
phone-only fields are null for USB. `wall_timestamp_utc` is the recorder processing time,
which can be later than capture/receipt when frames are buffered. Existing schema
`1.0`/`1.1` recordings remain readable and unchanged.

All collection settings are copied into session metadata, including experiment
notes, optional hardware geometry, receiver firmware provenance when flashed by
this application, source configuration and dependency versions. Existing board
firmware that was flashed outside this application cannot be verified from serial
CSI; its per-board firmware provenance remains unknown.

`configs/collection.yaml` holds versioned defaults (rate thresholds, bounded queue
sizes, stall timeout, disk threshold and synthetic controls). Rebuild containers
after changing it. `.env` holds host-specific paths and ports. Make records the
current project commit; set `PROJECT_GIT_COMMIT` yourself if launching Compose
directly. A commit ID alone does not identify uncommitted edits; session metadata also hashes the collector source and versioned configuration.

## Interrupted recordings

On restart, the hardware service marks interrupted sessions `incomplete` and
preserves `.filename.tmp` artifacts. CSI writes independent Zstandard frames at
least once per second so finished chunks can be inspected even if the last chunk
is truncated. Fragmented MP4 can retain flushed fragments; the most recent
fragment may be lost. A Parquet file without its final footer may be unreadable.
Recovery never invents missing timestamps or automatically promotes partial files.
Download artifacts from Sessions and inspect offline; preserve the original files.

```bash
make manifest             # rebuild root manifest from metadata
make logs                 # service logs
make down                 # graceful stop, data persists
```

## Tests

```bash
make test                 # unit + synthetic integration tests in Docker
make test-integration     # collection, synchronization, failures and API lifecycle
```

Tests use temporary directories. They verify both CSI formats, signed uint32
conversion, sequence wraps/gaps, configuration validation, synchronized multi-RX
recording, exact decoded video PTS/frame index agreement, manual stop, raw-file
immutability, hardware contention, camera/encoder failure, queue overflow, and
restart recovery, phone pairing/expiry, JPEG validation, phone timing, and safe
session removal.

The browser test uses Google Chrome with H.264 support in a disposable Docker image.
It drives board assignment, camera/CSI tests, preflight, manual
record/stop, and session playback/downloads against the full mock Compose stack:

```bash
make mock
make test-ui
# Also run phone pairing, recording, and removal in Chrome with a simulated camera:
make phone-cert PHONE_HOST=127.0.0.1
make phone
make test-ui PHONE_BASE_URL=https://127.0.0.1:8443
```

Opt-in read-only device tests live under `tests/hardware`; they never automatically
flash a board. Physical flashing and real radio/camera throughput require checking
on the actual devices. Software tests cannot establish hardware synchronization
accuracy or physical LED behavior.

## Troubleshooting

- **No packets / failed preflight:** start the transmitter; confirm channel 11,
  compatible firmware, RX 921600 baud, and matching MAC filtering. Use CSI test
  to inspect boot messages and malformed input.
- **Unexpected CSI format:** this parser supports the 25-field legacy and
  15-field compact CSV outputs plus CRC-protected binary v1. Unknown layouts are rejected
  with counts and retained original lines. Add a documented parser variant before
  collecting with different firmware.
- **Camera format/FPS mismatch:** by default, the collector disables the V4L2
  dynamic-exposure frame-rate control when available. This prevents supported
  cameras from silently dropping from 30 to 15 FPS for longer low-light exposure.
  Auto exposure stays enabled. Uncheck **Keep requested FPS** to allow dynamic
  frame rates. Tests show requested/measured FPS and negotiated format/exposure
  diagnostics. If the rate is still low, add light and review supported V4L2 formats, choose a supported
  width/height/FPS, and check USB bandwidth. The requested dimensions must match
  actual frames. This implementation requires even dimensions for H.264 YUV420.
- **Hardware busy:** close live preview, wait for the current job, or stop recording.
  Firmware writes cannot be cancelled mid-flash through the UI.
- **Incomplete recording:** inspect metadata errors and thread logs. If a driver
  thread cannot stop, restart the hardware service before another recording.
- **Docker permission denied:** use `DOCKER='sudo docker'` with Make, or configure
  Docker access according to your host policy.
- **Job remains after restart:** interrupted jobs become failed with a restart
  explanation. Raw data and logs remain available.

See [architecture](docs/architecture.md) for service boundaries and timing design.

See [validation notes](docs/validation.md) for the completed checks and hardware limits.
