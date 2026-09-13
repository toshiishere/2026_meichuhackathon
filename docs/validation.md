# Validation performed

Validated on 2026-09-11 in Linux Docker containers, with final browser and service checks on 2026-09-12.

Latest result (2026-09-12): 37 Python tests passed, 2 physical hardware tests
excluded, and both Google Chrome workflows passed (USB/synthetic collection and
phone HTTPS pairing/recording/removal).

- Real and synthetic hardware Docker images build successfully.
- Backend and React/TypeScript frontend production images build successfully.
- Mock Compose starts all three services with healthy checks at localhost:8080.
- Python unit and integration suite covers parser layouts, sequence accounting,
  partial serial reads over a pseudoterminal, USB identity, configuration validation,
  raw immutability, multi-receiver clock alignment, actual decoded MP4 PTS versus
  Parquet, manual stop, camera and encoder faults, bounded queue loss, interrupted
  sessions, persistent jobs, firmware cache invalidation, and backend endpoints.
- Playwright drives the full browser workflow, including explicit camera preview
  shutdown and hardware lease release, preflight, manual recording, raw artifact
  downloads, HTTP range requests, and session video playback.
- ESP-IDF v5.5 builds the preserved ESP32 CSI sender, ESP32-C3 CSI receiver, and
  ESP32-C3 WS2812 Blink firmware (GPIO 8) using the build-only CLI. No flash commands
  were executed and no connected board was modified.
- npm dependency audit reported no vulnerabilities in the resolved frontend tree.

No physical serial boards or V4L2 cameras were attached during the original checks. Radio packet rate, LED
wiring, USB driver behavior, camera modes, and real capture latency must be checked
on the actual hardware. Compilation and synthetic tests do not establish those.

The final frontend smoke check also verified that camera and CSI defaults match
`configs/collection.yaml`, no browser errors occurred, and the dashboard fits a
390-pixel mobile viewport without horizontal overflow. All three services remained
healthy after approximately ten hours running in synthetic mode.

## Phone camera and session removal update

- Phone end-to-end testing ran on a separate Compose project, ports 18080/18443,
  with separate temporary data and certificates. Existing user data was not used
  for recording or removal tests.
- Chrome used its fake media device with camera permission over HTTPS. The test
  verified pairing, frame delivery, fragment credential removal, 390-pixel phone
  layout, camera discovery/selection, automatic capture settings, preflight,
  at least 30 recorded phone frames alongside synthetic CSI, and MP4 loading.
- The same browser test checked typed removal confirmation, cancellation before
  submission, successful removal from the archive, and 404 responses for removed
  metadata and raw video. It also checked visible hardware-busy errors in the
  removal dialog and successful retry. Public phone routes rejected administrative
  API access.
- Python tests verified single-use pairing and expiry, invalid credentials, JPEG
  dimensions/payload limits, monotonic phone counters, bounded queue loss, and
  disconnect handling. Recorder integration verified separate phone capture times
  and collector host timestamps in Parquet schema 1.1.
- Removal tests cover hardware contention, active metadata, surviving acquisition
  threads, symlink rejection, manifest cleanup, stale backend index pruning, and
  preservation of recoverable files when disk cleanup fails.
- Serial tests confirm only ttyUSB0–9 and ttyACM0–9 are discovered and validated;
  ttyS ports and USB/ACM numbers above 9 are excluded. Stable USB identities remain.
- The updated local stack runs in real-hardware mode with all services healthy.
  Checksums confirmed all 12 pre-existing raw artifacts stayed unchanged. The
  configured phone address returned HTTPS 200 with lab-CA certificate verification.
- `PHONE_HOST` configuration accepts a DNS hostname or IPv4 address. Local CA and
  server certificate generation runs in Docker; both hostname and IP subject
  alternative names were verified. Private keys are not served.

Physical Android/iOS camera capture and certificate installation have not been
verified on a handset. Browser tests bypass certificate errors in their isolated
context; an actual phone must trust the lab CA. Arrival timestamps do not measure
or correct phone sensor, JPEG, or network latency.

## CSI throughput and camera correction (2026-09-13)

The recorded camera failures were `/dev/video0` (Logitech C270), 1280 × 720,
70 frames over a five-second test, approximately 15.0 FPS. V4L2 reported MJPG at
30 FPS, auto exposure in aperture-priority mode, and
`exposure_dynamic_framerate=1`. Disabling that control produced 140 frames and
29.58 FPS in the same test, with auto exposure still enabled. This is the control
that allows automatic exposure to vary FPS, per the [Linux camera control
reference](https://kernel.org/doc/html/v5.7/media/uapi/v4l/ext-ctrls-camera.html).
The collector now applies the fixed-frame-rate setting by default, exposes it in
the UI, and includes the negotiated mode and control results in diagnostics.
The C270's metadata-only `/dev/video1` is no longer offered as a camera.

Historical receiver tests showed about 65–66 packets/second, approximately
90–100 sequence gaps per five-second test, and hundreds of gain-log lines. A
representative CSV record was 1343 characters. The old callback formatted and
printed every sample and gain line inside the Wi-Fi task. The patch queues
unmodified CSI bytes for a worker and sends a 450-byte binary packet for 384
samples; host reads now operate on chunks. The sender's relative sleep was also
replaced with a periodic schedule. These remove identifiable software costs;
they are not a measured claim of 100 Hz over the physical radio.

Validation:

- 50 Python tests pass (two opt-in physical-device tests excluded), including
  raw binary metadata/sample/CRC preservation, chunk-boundary framing, corrupt
  frame recovery, bounded noise handling, 100 Hz pseudoterminal ingestion,
  synchronized binary CSI/video recording, firmware loss accounting, camera
  control application, and build-cache exclusion of generated files.
- A native C emitter using the firmware header produced exactly the same packet
  bytes and CRC as the independent Python packet fixture/decoder.
- ESP-IDF 5.5 compiled the queued/buffered binary receiver for ESP32, ESP32-C3,
  ESP32-C6, and ESP32-S3, and the periodic sender for ESP32.
- Frontend TypeScript and production Docker images build successfully. Both
  full-stack Chrome regressions pass: collection/playback and HTTPS phone
  pairing/recording/removal, using separate synthetic test data.
- The deployed app was tested in Chrome against the attached C270: **29.80 FPS,
  141 frames, 1280 × 720 MJPG**, requested/negotiated 30 FPS, PASS, no browser
  errors. The fixed-frame-rate checkbox and diagnostic result were verified.
- All three deployed services are healthy in real mode; running Python source
  matches the workspace. Checksums of all eight existing raw artifacts match
  their pre-update values.

Blink was explicitly excluded from this update. No board was flashed during these
checks. The receiver and sender changes take effect after the corresponding
firmware is installed through Hardware Setup. Camera correction was measured on
the attached webcam; firmware radio-rate improvements require a post-flash test.

### Follow-up: direct transport, external sender, and device registrations

The queued binary firmware still showed a USB output bottleneck: job
`ab7a99174c9141e68baebb3bda760d27` received 314 packets in 5.020 seconds
(62.55 Hz), with 97 firmware queue drops and no parse errors. The worker's
`fwrite` still used per-byte console VFS dispatch. Following the direct USB
driver approach in the pinned reference linked in [the transport document](csi-binary-v1.md),
frames now go directly to the selected USB/UART driver, with explicit short-write
retries and no mirrored runtime console output. The target stays at 100 Hz.

- ESP-IDF 5.5 builds pass for ESP32 UART, ESP32-C3 USB and UART, ESP32-C6 USB,
  and ESP32-S3 USB. Native C tests verify whole-frame writes and partial-write /
  backpressure recovery without interleaving frames.
- 59 Python tests pass (two physical-device tests excluded). New coverage checks
  receiver-only preflight, optional-sender multi-receiver/video timestamps,
  explicit sender identity validation, transport selection/cache configuration,
  and device-registration removal while retaining archived files.
- Three full-stack Chrome tests pass against isolated synthetic data: collection
  with no sender selected, HTTPS phone collection and session removal, and logical
  name mapping through port changes/disconnection/removal.

Sender selection now defaults to external power. Registered logical names show
current connections by USB identity; removal affects registration metadata only.
No ESP32 was connected at the end of validation, so the direct-output firmware
has not been flashed or physically rate-tested in this follow-up. The compiled
C3 native-USB cache key is `15e7fe4811b919db2369bb82`.
