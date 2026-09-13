# Firmware baseline and collection patches

Before implementation the local `/home/toshi/esp-csi` tree was inspected read-only.
Upstream commit: **8633d67152db2808f141cc1595970aa9cf406045**.
The local top-level `csi_send`, `csi_recv` and `blink` projects were untracked
working copies in that repository. Before the 2026-09-13 patches, both CSI `main/app_main.c` files matched
`examples/get-started` at that commit. Full SDK configurations and dependency lock
files are retained; build products and downloaded managed components are excluded.

`provenance.json` records SHA-256 for all preserved files. Original source and boards
were not modified during the original implementation. The vendored receiver now
uses queued binary output, and the sender uses a periodic schedule. See the
[transport specification and installation steps](../docs/csi-binary-v1.md).
The recorded upstream hashes remain the original baseline. Blink is unchanged.

| Setting | Preserved value |
|---|---|
| ESP-IDF | 5.5.0 (generated SDK headers and dependency locks) |
| Official image | `espressif/idf:v5.5` |
| Sender target | esp32 |
| Receiver target | esp32c3 |
| Rate | 100 Hz (`CONFIG_SEND_FREQUENCY`) |
| Channel / bandwidth | 11 / HT40 |
| Sender MAC filter | `1a:00:00:00:00:00` |
| Receiver serial | 921600 baud |
| TX sequence | uint32 counter carried in ESP-NOW payload |

The updated receiver preserves original int8 CSI samples and gain metadata in
CRC-protected binary records. The host also accepts the original CSV firmware,
whose sample values were gain-compensated int16. The stored representation is
explicit in each new CSV row, and the original binary frame or CSV line is retained.
TX sequences and ESP local timestamps remain uint32 values.

Building for the original target preserves the saved full SDK config. Selecting a
different target uses the project's `sdkconfig.defaults`; it is a distinct build
configuration and must be verified on actual hardware. Firmware build success does
not prove the target board has the selected LED pin or usable camera/radio timing.

Blink supports GPIO and WS2812 LEDs, adjustable GPIO and active-low GPIO output.
No LED wiring can be determined reliably from chip family alone.

Sources: [Espressif ESP-CSI at the preserved commit](https://github.com/espressif/esp-csi/tree/8633d67152db2808f141cc1595970aa9cf406045/examples/get-started),
[official ESP-IDF Docker guide](https://docs.espressif.com/projects/esp-idf/en/v5.5/esp32/api-guides/tools/idf-docker-image.html).

Build-only CLI, with no physical board required (run using the real hardware image):

```bash
docker compose run --rm --no-deps hardware-service python scripts/build_firmware.py csi-send --target esp32
docker compose run --rm --no-deps hardware-service python scripts/build_firmware.py csi-recv --target esp32c3
docker compose run --rm --no-deps hardware-service python scripts/build_firmware.py blink --target esp32c3 --gpio 8 --led-type rgb
```
