# CSI binary transport v1

The 2026-09-13 receiver patch replaces per-sample printing and gain logging in the
Wi-Fi callback. Espressif explicitly recommends queuing CSI for a lower-priority
task: [Wi-Fi CSI guidance](https://docs.espressif.com/projects/esp-idf/en/v5.5.4/esp32/api-guides/wifi.html).

The callback validates the source MAC and payload/CSI length, takes a slot from a
32-entry fixed pool without waiting, copies the radio metadata and CSI bytes, and
queues the slot. It does no allocation, output, or gain compensation. The worker
computes gain metadata, serializes a record, and sends the frame directly to the
selected UART or native USB driver. This follows the bulk USB output approach in
[TryTwoTop/esp32c5-csi-keystroke](https://github.com/TryTwoTop/esp32c5-csi-keystroke/tree/8999ce8379266d5efb8b7cb3e73e6bdcd1db323d).
The former `fwrite` console path dispatched individual bytes through VFS and could
mirror output to another console. Binary data now bypasses console VFS entirely;
runtime logs are suppressed during capture. Short writes are retried from their
exact remaining offset before sending another frame. Full queues drop new packets and report cumulative
counts on subsequent records. Invalid/oversized input is counted, never truncated.
The sample limit is 1024 bytes; it is not a fixed subcarrier count.

The sender uses a periodic FreeRTOS schedule at 100 Hz instead of sleeping 10 ms
after each send. If it overruns, it restarts the schedule without catch-up bursts.
Radio/USB performance must still be measured after installing the new firmware.

## Wire layout

All multibyte fields are little endian. The C layout is in
`firmware/esp-csi/csi_recv/main/csi_wire.h`; the host decoder is
`apps/hardware_service/app/csi_wire.py`. IDF-specific bitfields are never put on
the wire.

| Bytes | Meaning |
|---|---|
| 0–3 | Magic `a5 43 53 49` |
| 4 | Version `01` |
| 5–6 | uint16 body length (55-byte metadata plus sample count) |
| 7 onward | Metadata, then original signed int8 samples |
| Last 4 | Standard CRC-32/IEEE of version, body length, and body (bytes 4 through the last sample) |

The 55-byte packed metadata contains, in order:

| Type | Fields |
|---|---|
| 5 × uint32 | TX sequence, ESP local timestamp, matching callbacks received, queue drops, invalid inputs |
| 6 bytes | Source MAC |
| 3 × int8 | RSSI, noise floor, FFT gain |
| uint8 | AGC gain |
| float32 | Gain compensation factor |
| 2 × uint16 | Signal length, CSI byte/sample count |
| 17 × uint8 | Rate, signal mode, MCS, bandwidth, smoothing, not-sounding, aggregation, STBC, FEC coding, SGI, AMPDU count, channel, secondary channel, antenna, RX format, first-word-invalid, PHY layout |

PHY layout 0 carries legacy metadata (ESP32/C3/S3); layout 1 carries the compact
metadata available on C6. Unavailable fields remain empty in host CSV rows.
A 384-sample record is **450 bytes**, versus approximately **1343 characters** in
the observed legacy CSV, plus its separate gain log. At 100 records/second this
uses 45 kB/s, below the 92.16 kB/s payload limit of a 921600-baud 8N1 UART.
This size estimate does not prove radio delivery or native USB throughput.

## Storage and synchronization

The collector accepts mixed boot text, legacy CSV, compact CSV, and binary v1.
It reads available serial bytes in chunks instead of calling pyserial's
byte-at-a-time `read_until`. Each completed frame retains the host monotonic
receipt time of its final chunk before parsing, compression, or disk writing.
Frames completed in the same chunk can share a timestamp; no artificial timing
is assigned within a buffered chunk. CRC failures are counted and their original
bytes are preserved in serial logs. Framing recovers at subsequent packets.

Binary rows still go into `raw/csi_<receiver>.csv.zst`, with these additive fields:

- `firmware_layout=binary_v1`, `sample_representation=raw_int8`.
- `data`: all original signed int8 samples, imaginary then real, including any
  hardware-invalid first word. **No gain multiplication** is performed.
- `compensate_gain`, `gain_agc`, `gain_fft`: gain context, not applied transforms.
- `raw_binary_base64`: the entire original wire frame, including metadata and CRC.
- `firmware_received_total`, `firmware_queue_drops_total`, `firmware_invalid_total`:
  cumulative uint32 values observed by the callback. These are snapshots, not a
  final device-wide loss tally if a board disconnects before its next report.
- `raw_line` is empty for binary rows; it remains the exact original line for CSV.

CSV rows retain their original gain-compensated int16 values and now explicitly
state `sample_representation=gain_compensated_int16`. Consumers must check this
field/layout before combining sample values across firmware versions. Existing
recordings are not rewritten. Session metadata stays at schema 1.0; the raw CSI
CSV header gains the fields above. Phone video-frame schema remains 1.1.

Test and session statistics show firmware queue losses since their first packet,
separately from host queue losses and transmitter sequence gaps. Counter resets
are reported without inventing huge losses. Firmware queue loss during recording
marks quality degraded; no missing packet is filled in.

## Install

Build/rebuild and flash **CSI Receiver** on each receiver from Hardware Setup,
using its correct chip target. Rebuild/flash **CSI Sender** on the sender to get
the periodic schedule. Existing CSV receiver firmware continues to work with the
updated collector. Host updates alone do not install firmware on boards.

**CSI output connection** defaults to automatic: native USB for supported chips
on ttyACM / USB JTAG serial, UART for USB-to-UART adapters and original ESP32.
Override it to match the board connector if automatic detection is unsuitable.
The resolved transport is part of the firmware cache key. CLI builds can specify
`--csi-transport usb` or `--csi-transport uart`. The target remains **100 Hz**.

The original upstream provenance hashes remain in `firmware/provenance.json`.
These are baseline hashes, not claims that the patched source is unchanged.
New firmware build records include the full current source hash and cache key.
Generated `build/` and `managed_components/` directories are excluded from the
hash and cache copy; dependency lockfiles remain part of the build input.
