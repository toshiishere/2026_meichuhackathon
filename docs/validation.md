# Validation performed

Validated on 2026-09-11 in Linux Docker containers, with final browser and service checks on 2026-09-12.

Result: 28 Python tests passed, 2 physical hardware tests excluded, and the full Google Chrome recording/playback test passed.

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

No physical serial boards or V4L2 cameras were attached. Radio packet rate, LED
wiring, USB driver behavior, camera modes, and real capture latency must be checked
on the actual hardware. Compilation and synthetic tests do not establish those.

The final frontend smoke check also verified that camera and CSI defaults match
`configs/collection.yaml`, no browser errors occurred, and the dashboard fits a
390-pixel mobile viewport without horizontal overflow. All three services remained
healthy after approximately ten hours running in synthetic mode.
