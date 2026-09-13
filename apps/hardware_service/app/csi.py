"""CSV layouts verified against the preserved ESP-CSI source, not guessed by width.

ESP-CSI emits imaginary then real values, with gain-compensated int16 samples.
The raw data string is retained verbatim. No subcarriers or invalid first words
are removed in acquisition. `id`/`seq` is the sender uint32 counter in this source.
"""

import csv
import json
import re
from .csi_wire import MAGIC, decode_binary

LEGACY = (
    "type id mac rssi rate sig_mode mcs bandwidth smoothing not_sounding "
    "aggregation stbc fec_coding sgi noise_floor ampdu_cnt channel secondary_channel "
    "local_timestamp ant sig_len rx_format len first_word data"
).split()
COMPACT = (
    "type seq mac rssi rate noise_floor fft_gain agc_gain channel local_timestamp "
    "sig_len rx_format len first_word data"
).split()
FIELDS = [
    "host_timestamp_ns",
    "wall_timestamp_utc",
    "receiver_id",
    "port",
    "tx_seq",
    "esp_local_timestamp",
    "firmware_layout",
    *dict.fromkeys([*LEGACY, *COMPACT]),
    "raw_line",
    "compensate_gain",
    "gain_agc",
    "gain_fft",
    "sample_representation",
    "raw_binary_base64",
    "firmware_received_total",
    "firmware_queue_drops_total",
    "firmware_invalid_total",
]
GAIN = re.compile(r"compensate_gain ([0-9.eE+-]+), agc_gain (-?\d+), fft_gain (-?\d+)")


def parse_csi(line: str | bytes):
    if isinstance(line, bytes):
        if line.startswith(MAGIC):
            return decode_binary(line)
        line = line.decode("utf-8", errors="replace")
    if not line.startswith("CSI_DATA,"):
        return None
    try:
        row = next(csv.reader([line], strict=True))
        fields = {25: LEGACY, 15: COMPACT}.get(len(row))
        if fields is None:
            raise ValueError(
                f"Unsupported CSI field count {len(row)}; expected 25 or 15"
            )
        record = dict(zip(fields, row))
        if not re.fullmatch(r"(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}", record["mac"]):
            raise ValueError("Invalid MAC address")
        for k in fields:
            if k not in {"type", "mac", "data"}:
                record[k] = int(record[k])
        samples = json.loads(record["data"])
        if not isinstance(samples, list) or not samples or len(samples) % 2:
            raise ValueError("CSI data needs a nonempty even-length array")
        if any(type(v) is not int or not -32768 <= v <= 32767 for v in samples):
            raise ValueError("CSI values must be int16 integers")
        if len(samples) != record["len"]:
            raise ValueError("CSI length does not match emitted array")
        if record["first_word"] not in (0, 1):
            raise ValueError("Invalid first_word flag")
        seq = record.get("seq", record.get("id"))
        if not -(2**31) <= seq < 2**32:
            raise ValueError("Sequence outside uint32 range")
        # Upstream printf uses %d for uint32. Keep original field as well.
        record["tx_seq"] = seq % 2**32
        record["esp_local_timestamp"] = record["local_timestamp"] % 2**32
        record["firmware_layout"] = "legacy_25" if len(row) == 25 else "compact_15"
        record["raw_line"] = line
        record["sample_representation"] = "gain_compensated_int16"
        return record
    except (csv.Error, TypeError, KeyError, json.JSONDecodeError) as e:
        raise ValueError(f"Malformed CSI: {e}") from e


class SequenceTracker:
    """Count forward missing IDs, duplicates, backwards/reset events and uint32 wraps.

    A backwards event is ambiguous (reset or reordering); expose it without
    inventing a giant loss. Reset the baseline so subsequent counts remain useful.
    """

    def __init__(self):
        self.last = None
        self.gaps = self.duplicates = self.backwards = self.wraps = 0

    def update(self, seq):
        if self.last is not None:
            delta = (seq - self.last) % 2**32
            if delta == 0:
                self.duplicates += 1
            elif delta < 2**31:
                self.gaps += delta - 1
                self.wraps += int(seq < self.last)
            else:
                self.backwards += 1
        self.last = seq

    def snapshot(self):
        return dict(
            latest_tx_seq=self.last,
            sequence_gaps=self.gaps,
            duplicates=self.duplicates,
            backwards_or_resets=self.backwards,
            wraps=self.wraps,
        )


class TransportTracker:
    """Delta firmware counters within this acquisition, excluding prior losses."""

    def __init__(self):
        self.last = {}
        self.values = {}

    def update(self, record):
        for key in ("firmware_queue_drops", "firmware_invalid"):
            total = record.get(key + "_total")
            if total is None:
                continue
            if key in self.last:
                delta = (total - self.last[key]) % 2**32
                if delta < 2**31:
                    self.values[key] = self.values.get(key, 0) + delta
                else:
                    self.values["firmware_counter_resets"] = (
                        self.values.get("firmware_counter_resets", 0) + 1
                    )
            else:
                self.values[key] = 0
            self.last[key] = total
        return self.values.copy()
