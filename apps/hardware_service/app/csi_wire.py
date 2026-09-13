"""Lossless binary CSI v1 decoding and mixed boot-text/binary serial framing."""

import base64
import json
import math
import struct
import zlib

MAGIC = b"\xa5CSI"
HEADER = struct.Struct("<4sBH")
META = struct.Struct("<5I6s3bBf2H17B")
MAX_SAMPLES = 1024
MAX_BODY = META.size + MAX_SAMPLES
PHY_FIELDS = "rate sig_mode mcs bandwidth smoothing not_sounding aggregation stbc fec_coding sgi ampdu_cnt channel secondary_channel ant rx_format first_word layout".split()


def decode_binary(raw):
    if len(raw) < HEADER.size:
        raise ValueError("Truncated binary CSI header")
    magic, version, length = HEADER.unpack_from(raw)
    if magic != MAGIC or version != 1 or not META.size < length <= MAX_BODY:
        raise ValueError("Unsupported binary CSI version or length")
    if len(raw) != HEADER.size + length + 4:
        raise ValueError("Truncated binary CSI frame")
    (expected,) = struct.unpack_from("<I", raw, len(raw) - 4)
    if zlib.crc32(raw[4:-4]) != expected:
        raise ValueError("Binary CSI CRC mismatch")
    values = META.unpack_from(raw, HEADER.size)
    (
        seq,
        stamp,
        received,
        dropped,
        invalid,
        mac,
        rssi,
        noise,
        fft,
        agc,
        gain,
        sig_len,
        count,
        *phy,
    ) = values
    fields = dict(zip(PHY_FIELDS, phy))
    layout = fields.pop("layout")
    if count == 0 or count % 2 or count > MAX_SAMPLES or META.size + count != length:
        raise ValueError("Binary CSI sample length mismatch")
    if (
        layout not in (0, 1)
        or fields["first_word"] not in (0, 1)
        or not math.isfinite(gain)
        or gain <= 0
    ):
        raise ValueError("Invalid binary CSI metadata")
    if layout == 1:
        fields = {
            k: v
            for k, v in fields.items()
            if k in {"rate", "channel", "rx_format", "first_word"}
        }
    samples = struct.unpack_from(f"<{count}b", raw, HEADER.size + META.size)
    return dict(
        type="CSI_DATA",
        **fields,
        **({"id": seq} if layout == 0 else {"seq": seq}),
        mac=":".join(f"{b:02x}" for b in mac),
        rssi=rssi,
        noise_floor=noise,
        local_timestamp=stamp,
        sig_len=sig_len,
        len=count,
        tx_seq=seq,
        esp_local_timestamp=stamp,
        firmware_layout="binary_v1",
        data=json.dumps(samples, separators=(",", ":")),
        raw_line="",
        raw_binary_base64=base64.b64encode(raw).decode(),
        sample_representation="raw_int8",
        compensate_gain=gain,
        gain_agc=agc,
        gain_fft=fft,
        **({"agc_gain": agc, "fft_gain": fft} if layout == 1 else {}),
        firmware_received_total=received,
        firmware_queue_drops_total=dropped,
        firmware_invalid_total=invalid,
    )


class SerialFramer:
    def __init__(self, max_text_bytes=32768):
        self.buffer = bytearray()
        self.max_text_bytes = max_text_bytes

    def feed(self, chunk, stamp):
        self.buffer.extend(chunk)
        result = []
        while self.buffer:
            if self.buffer.startswith(MAGIC):
                if len(self.buffer) < HEADER.size:
                    break
                _, version, size = HEADER.unpack_from(self.buffer)
                if version != 1 or not META.size < size <= MAX_BODY:
                    result.append((bytes(self.buffer[: HEADER.size]), stamp))
                    del self.buffer[: HEADER.size]
                    continue
                end = HEADER.size + size + 4
                if len(self.buffer) < end:
                    break
                candidate = bytes(self.buffer[:end])
                if (
                    zlib.crc32(candidate[4:-4])
                    != struct.unpack_from("<I", candidate, end - 4)[0]
                ):
                    # A damaged length or interrupted packet may contain the next
                    # valid frame. Resume there, retaining rejected bytes for logs.
                    next_magic = self.buffer.find(MAGIC, 1)
                    if next_magic >= 0:
                        end = next_magic
                result.append((bytes(self.buffer[:end]), stamp))
                del self.buffer[:end]
                continue
            magic = self.buffer.find(MAGIC)
            newline = self.buffer.find(b"\n")
            if newline >= 0 and (magic < 0 or newline < magic):
                result.append((bytes(self.buffer[:newline]).rstrip(b"\r"), stamp))
                del self.buffer[: newline + 1]
            elif magic > 0:
                result.append((bytes(self.buffer[:magic]), stamp))
                del self.buffer[:magic]
            elif len(self.buffer) > self.max_text_bytes:
                # Retain a possible split magic prefix, bound arbitrary noise.
                result.append((bytes(self.buffer[:-3]), stamp))
                del self.buffer[:-3]
            else:
                break
        return result
