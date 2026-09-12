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
