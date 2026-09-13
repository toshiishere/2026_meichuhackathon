import base64
import json
import os
import pty
import struct
import threading
import time
import zlib

import pytest
from apps.hardware_service.app.csi import parse_csi, TransportTracker
from apps.hardware_service.app.csi_wire import MAGIC, HEADER, META, SerialFramer
from apps.hardware_service.app import sources


def binary_frame(seq=123, drops=10, invalid=0, samples=bytes(range(256)), layout=0):
    # Independent field values include signs, uint32 boundaries, and a gain that
    # must stay metadata rather than changing the original sample bytes.
    meta = META.pack(
        seq,
        0xFFFFFFF0,
        seq + 1,
        drops,
        invalid,
        bytes.fromhex("1a0000000000"),
        -43,
        -95,
        -7,
        18,
        1.25,
        40,
        len(samples),
        11,
        1,
        2,
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        3,
        11,
        2,
        0,
        1,
        1,
        layout,
    )
    data = HEADER.pack(MAGIC, 1, len(meta) + len(samples)) + meta + samples
    return data + struct.pack("<I", zlib.crc32(data[4:]))


def test_binary_preserves_raw_samples_gains_metadata_and_original_bytes():
    wire = binary_frame()
    record = parse_csi(wire)
    assert record["tx_seq"] == 123 and record["esp_local_timestamp"] == 0xFFFFFFF0
    assert record["rssi"] == -43 and record["noise_floor"] == -95
    assert record["gain_fft"] == -7 and record["compensate_gain"] == 1.25
    assert record["first_word"] == 1 and record["mcs"] == 2
    assert json.loads(record["data"]) == list(range(128)) + list(range(-128, 0))
    assert record["sample_representation"] == "raw_int8"
    assert base64.b64decode(record["raw_binary_base64"]) == wire
    assert record["raw_line"] == ""
    compact = parse_csi(binary_frame(layout=1))
    assert compact["seq"] == 123 and "mcs" not in compact
    assert compact["agc_gain"] == 18


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 64, 8192])
def test_binary_framing_handles_split_headers_text_and_payload_delimiters(chunk_size):
    first = binary_frame(samples=b"\n\r\x00\xa5CSI\xff")
    second = binary_frame(seq=124)
    stream = b"boot message\r\n" + first + b"gain log\n" + second
    framer = SerialFramer()
    output = []
    for i in range(0, len(stream), chunk_size):
        output += framer.feed(stream[i : i + chunk_size], i + 1)
    assert [raw for raw, stamp in output] == [
        b"boot message",
        first,
        b"gain log",
        second,
    ]
    assert [stamp for raw, stamp in output] == sorted(stamp for raw, stamp in output)
    assert not framer.buffer


def test_binary_corruption_recovery_and_bounds():
    bad = bytearray(binary_frame())
    bad[-1] ^= 0x80
    with pytest.raises(ValueError, match="CRC"):
        parse_csi(bytes(bad))
    good = binary_frame(seq=124)
    framer = SerialFramer()
    output = framer.feed(bytes(bad) + good, 100)
    assert output == [(bytes(bad), 100), (good, 100)]
    truncated = binary_frame()[:100]
    output = SerialFramer().feed(truncated + good, 200)
    assert output[-1][0] == good
    with pytest.raises(ValueError):
        parse_csi(binary_frame(samples=b"abc"))
    with pytest.raises(ValueError):
        parse_csi(binary_frame(layout=2))
    bounded = SerialFramer(32)
    bounded.feed(b"x" * 10000, 1)
    assert len(bounded.buffer) <= 32


def test_real_serial_binary_ingest_keeps_up_with_100hz(monkeypatch):
    master, slave = pty.openpty()
    monkeypatch.setattr(sources, "MODE", "real")
    monkeypatch.setattr(sources, "validate_port", lambda _: {"identity": "test"})
    source = sources.SerialSource(os.ttyname(slave), 921600)
    stop = threading.Event()

    def transmit():
        start = time.monotonic()
        for seq in range(120):
            if stop.wait(max(0, start + seq * 0.01 - time.monotonic())):
                return
            raw = binary_frame(seq=seq, samples=bytes(range(256)) + bytes(range(128)))
            os.write(master, raw[:5])
            os.write(master, raw[5:])

    thread = threading.Thread(target=transmit)
    thread.start()
    received = []
    try:
        deadline = time.monotonic() + 3
        while len(received) < 120 and time.monotonic() < deadline:
            value = source.read(stop)
            if value:
                received.append((parse_csi(value[0]), value[1]))
        assert [r["tx_seq"] for r, s in received] == list(range(120))
        fps = (len(received) - 1) * 1e9 / (received[-1][1] - received[0][1])
        assert 90 < fps < 110
    finally:
        stop.set()
        thread.join()
        source.close()
        os.close(master)
        os.close(slave)


def test_firmware_loss_counters_exclude_history_and_handle_wrap_reset():
    tracker = TransportTracker()
    assert (
        tracker.update({"firmware_queue_drops_total": 100})["firmware_queue_drops"] == 0
    )
    assert (
        tracker.update({"firmware_queue_drops_total": 103})["firmware_queue_drops"] == 3
    )
    assert (
        tracker.update({"firmware_queue_drops_total": 0})["firmware_counter_resets"]
        == 1
    )
    assert (
        tracker.update({"firmware_queue_drops_total": 2})["firmware_queue_drops"] == 5
    )
