# CSI Collection Lab

A Docker-first web application for **synchronized WiFi CSI and USB camera collection**.
Configure ESP32 boards, verify receiver/camera health, record multiple receivers
and a camera together, and inspect immutable raw sessions.

This implements the collection scope in `prompt` (milestones 1–6). Video labeling,
HAR, training, inference, and downstream feature/window dataset generation are
intentionally excluded. There are no dependencies or services for those tasks.

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

1. Select `synthetic://tx`, assign `tx_main`, role **CSI Sender**, and save.
2. Select `synthetic://rx0`, assign `rx_left`, role **CSI Receiver**, and save.
3. Optionally assign `synthetic://rx1` as `rx_right`.
4. Run **CSI test / monitor** on a receiver. Open the job result and logs.
5. Open the camera preview, then close it and run **Test camera**.
6. In **Data Collection**, select the sender, receivers, and camera. Enter a unique
   session ID and experiment details. Run preflight, then **Start recording**.
7. **Stop recording** whenever desired, or set an optional duration. Wait for
   `complete`, then open **Sessions** for playback, metadata, quality, and downloads.

Synthetic recording defaults to 100 CSI packets/sec/receiver and 1% seeded packet
loss. Loss and seed are configurable. Synthetic cameras render a moving marker,
frame number and monotonic time. Sessions explicitly record `hardware_mode`.
Synthetic mode never reports a physical probe or firmware flash as successful;
flashing is unavailable there.

## Real hardware

Required: an ESP32 sender and one or more ESP32 receivers running the preserved
Espressif CSI projects, USB serial connections, and a Linux USB/UVC camera.

```bash
make down
make up
# or, without Make:
docker compose up -d --build
```

The normal Compose file uses **`espressif/idf:v5.5`**, matching the ESP-IDF 5.5.0
headers and dependency locks in your existing working projects. The first real
hardware image build downloads the ESP-IDF toolchains and can take several minutes.
See [firmware provenance](firmware/README.md) and
[Espressif's official Docker documentation](https://docs.espressif.com/projects/esp-idf/en/v5.5/esp32/api-guides/tools/idf-docker-image.html).

The working source was found in `/home/toshi/esp-csi` and copied into this repo;
no host path dependency remains. The original repository and boards were not
modified. Receiver output: **921600 baud**. Sender: **100 Hz**, **channel 11**,
**HT40**, source MAC `1a:00:00:00:00:00`. See [firmware/README.md](firmware/README.md)
for the preserved commit, SDK configs, targets and hashes.

### Identify and flash boards

Refresh Hardware Setup after reconnecting USB devices. Stable `/dev/serial/by-id`
paths and USB identities preserve assignments across tty renumbering. Devices
without unique USB serials fall back to stable paths, USB location, then tty path;
verify these assignments again after moving USB sockets.

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

## Recording and synchronization

Preflight verifies all selected devices, identity consistency, actual parsable
CSI arrival and rate, requested camera resolution and measured FPS, writable
storage, and at least 1 GiB free disk (configurable). It runs again on every start.
No override bypasses failed checks. The sender is discovered; its transmission is
verified indirectly through received CSI packets, not by opening its serial port.

All devices and outputs initialize before a shared acquisition gate opens.
CSI packets and camera frames receive **`time.monotonic_ns()`** timestamps in the
same service process immediately after complete line receipt / camera read.
ESP local timestamps are microseconds (uint32, wrapping); UTC time is only descriptive. Serial queues, compression, video encoding, and HTTP
requests do not define acquisition timestamps.

Use `tx_seq` to align receivers. Each raw row also preserves ESP local timestamp,
original `id`/`seq`, MAC, RSSI, all emitted PHY fields, gain context when emitted,
`first_word`, exact CSI array string, and original CSV line. Arrays are not clipped,
converted to amplitude, or assigned a fixed subcarrier count. The source emits
**imaginary then real** values, including gain-compensated int16 samples.

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

## Data layout

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
      video_frames.parquet         # frame index, PTS/time base, host/UTC time
    logs/
      collection.log
      serial_rx_left.jsonl         # boot/gain lines and rejected records
      serial_rx_right.jsonl
      <failing-thread>.log         # traceback when an operation fails
```

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
restart recovery.

The browser test uses Google Chrome with H.264 support in a disposable Docker image.
It drives board assignment, camera/CSI tests, preflight, manual
record/stop, and session playback/downloads against the full mock Compose stack:

```bash
make mock
make test-ui
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
  15-field compact outputs in the preserved source. Unknown layouts are rejected
  with counts and retained original lines. Add a documented parser variant before
  collecting with different firmware.
- **Camera format/FPS mismatch:** review supported V4L2 formats, choose a supported
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
