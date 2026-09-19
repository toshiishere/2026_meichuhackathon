"""Build timestamp-aligned single-link ResNet18 windows without changing raw data."""

import csv
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.io as sio
import zstandard

from apps.common.storage import atomic_json
from .timeline import Timeline

LABELS = ["Static", "Walking", "Sitting", "Standing", "Falling"]
INDICES = {0: np.r_[1:27, 38:64], 1: np.r_[6:32, 33:59]}


def load_labels(path, timeline):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not {"start_time", "end_time", "label"}.issubset(reader.fieldnames or []):
            raise ValueError(
                "Labels need start_time,end_time,label (times in video milliseconds)"
            )
        rows = []
        for i, row in enumerate(reader):
            a, b = float(row["start_time"]) / 1000, float(row["end_time"]) / 1000
            if (
                not np.isfinite([a, b]).all()
                or a < timeline.pts[0]
                or b > timeline.pts[-1] + 1e-6
                or b <= a
                or row["label"] not in LABELS
            ):
                raise ValueError(f"Invalid label interval at CSV row {i + 2}")
            b = min(b, timeline.pts[-1])
            if rows and a < rows[-1][1]:
                raise ValueError(
                    "Label intervals overlap or are out of order; review action_results.csv"
                )
            rows.append((a, b, row["label"]))
    if not rows:
        raise ValueError(
            "No action labels were produced. Check visibility and recording length."
        )
    return rows


def packet_amplitude(row):
    """The same RF layout validation and 52 amplitudes for training and deployment."""
    bw, length = int(row["bandwidth"]), int(row["len"])
    if (
        bw not in INDICES
        or int(row["sig_mode"]) != 1
        or int(row["stbc"]) != 0
        or length != {0: 256, 1: 384}[bw]
    ):
        raise ValueError("unsupported_layout")
    if int(row["first_word"]) != 0:
        raise ValueError("invalid_first_word")
    values = json.loads(row["data"]) if isinstance(row["data"], str) else row["data"]
    if (
        not isinstance(values, list)
        or len(values) != length
        or any(type(v) is not int or not -32768 <= v <= 32767 for v in values)
    ):
        raise ValueError("malformed")
    a = np.asarray(values, dtype=np.float32).reshape(-1, 2)
    return np.hypot(a[:, 0], a[:, 1])[INDICES[bw]]


def packet_rows(path):
    """Stream collector rows without decompressing or loading a whole recording."""
    with path.open("rb") as raw:
        stream = (
            zstandard.ZstdDecompressor().stream_reader(raw)
            if path.suffix == ".zst"
            else raw
        )
        with io.TextIOWrapper(stream, encoding="utf-8-sig") as text:
            reader = csv.DictReader(text)
            required = {
                "host_timestamp_ns",
                "bandwidth",
                "sig_mode",
                "stbc",
                "len",
                "first_word",
                "data",
            }
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"{path.name}: need a timestamped collector CSV/.csv.zst; standalone wire binary has no host clock"
                )
            yield from reader


def normalize_window(amplitudes):
    """Per-window global z-score, as the training loader applies it.

    Mirrors load_mat_amplitude in csi_model/finetune/session_tools/finetune.py;
    keep the two together. scipy's MAT loader hands training column-major
    arrays, so reducing in that order keeps a live window numerically identical
    to the ones the saved model was trained on.
    """
    x = np.asfortranarray(amplitudes)
    return ((x - np.mean(x)) / (np.std(x) + 1e-8)).astype(np.float32)


def resample_window(stamps, amplitudes, lo, hi, size):
    """Reject gaps/low coverage, then interpolate onto the model's uniform clock."""
    stamps = np.asarray(stamps, dtype=np.int64)
    amplitudes = np.asarray(amplitudes, dtype=np.float32)
    left, right = max(0, np.searchsorted(stamps, lo, side="right") - 1), min(
        len(stamps), np.searchsorted(stamps, hi) + 1
    )
    stamps, amplitudes = stamps[left:right], amplitudes[left:right]
    if (
        len(stamps) < max(3, size * 0.7)
        or stamps[0] > lo
        or stamps[-1] < hi
        or np.max(np.diff(stamps)) > 200_000_000
    ):
        return None
    grid = np.linspace(0, hi - lo, size, endpoint=False)
    return np.stack(
        [np.interp(grid, stamps - lo, amplitudes[:, c]) for c in range(52)], axis=1
    ).astype(np.float32)


def decode_packets(path):
    times, amplitudes, rejected = [], [], Counter()
    for row in packet_rows(path):
        try:
            stamp = int(row["host_timestamp_ns"])
            amp = packet_amplitude(row)
            times.append(stamp)
            amplitudes.append(amp)
        except (ValueError, TypeError, KeyError) as error:
            reason = str(error)
            rejected[
                (
                    reason
                    if reason in {"unsupported_layout", "invalid_first_word"}
                    else "malformed"
                )
            ] += 1
    if len(times) < 3:
        raise ValueError(
            f"{path.name}: insufficient supported CSI packets; rejected={dict(rejected)}. Expected non-STBC HT20/HT40 L-LTF with valid first word."
        )
    order = np.argsort(times, kind="stable")
    times, amplitudes = (
        np.asarray(times, dtype=np.int64)[order],
        np.asarray(amplitudes)[order],
    )
    unique = np.r_[True, np.diff(times) > 0]
    rejected["duplicate_timestamps"] += int((~unique).sum())
    return times[unique], amplitudes[unique], dict(rejected)


def build_dataset(
    session, labels_path, out, window_seconds=2.0, sample_rate_hz=100, overlap=0.5
):
    timeline = Timeline(session / "raw/video_frames.parquet")
    labels = load_labels(labels_path, timeline)
    files = {
        p.name.removesuffix(".zst"): p
        for p in sorted((session / "raw").glob("csi_*.csv*"))
        if p.name.endswith((".csv", ".csv.zst"))
    }
    if not files:
        raise ValueError(
            "Session has no timestamped raw/csi_*.csv.zst or .csv recordings"
        )
    out.mkdir(parents=True, exist_ok=False)
    manifest, counts, skips, packets = [], Counter(), Counter(), {}
    size = round(window_seconds * sample_rate_hz)
    if size < 16 or window_seconds <= 0 or not 0 <= overlap < 1:
        raise ValueError("Invalid window size, overlap or sample rate")
    for name, path in files.items():
        receiver = name.removeprefix("csi_").removesuffix(".csv")
        ts, amps, rejected = decode_packets(path)
        packets[receiver] = dict(accepted=len(ts), rejected=rejected)
        for interval, (a, b, label) in enumerate(labels):
            # Only wholly labeled windows; never invent labels for surrounding context.
            if b - a < window_seconds - 1e-6:
                skips["short_interval"] += 1
                continue
            starts = list(
                np.arange(a, b - window_seconds + 1e-6, window_seconds * (1 - overlap))
            )
            if b - window_seconds - starts[-1] > 0.1:
                starts.append(b - window_seconds)
            for start in starts:
                end = min(start + window_seconds, b)
                if not timeline.contiguous(start, end):
                    skips["camera_gap"] += 1
                    continue
                lo, hi = timeline.host_ns(start), timeline.host_ns(end)
                amp = resample_window(ts, amps, lo, hi, size)
                if amp is None:
                    skips["csi_coverage"] += 1
                    continue
                folder = out / label
                folder.mkdir(exist_ok=True)
                filename = f"{receiver}-{len(manifest):06d}.mat"
                sio.savemat(folder / filename, {"CSIamp": amp})
                manifest.append(
                    dict(
                        session_id=session.name,
                        receiver=receiver,
                        label=label,
                        filename=filename,
                        interval_id=f"{session.name}:{interval}",
                        window_start_s=float(start),
                        window_end_s=float(end),
                        capture_start_ns=lo,
                        capture_end_ns=hi,
                        n_samples=size,
                    )
                )
                counts[label] += 1
        print(
            f"Preprocessed {receiver}: {len(ts)} packets; rejected {rejected}",
            flush=True,
        )
    if not manifest:
        raise ValueError(
            f"No usable CSI windows. Labels must contain at least {window_seconds}s of one action with camera/CSI coverage. Skips: {dict(skips)}"
        )
    with (out / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    summary = dict(
        windows=len(manifest),
        class_counts=dict(counts),
        skipped=dict(skips),
        packets=packets,
        clock_column=timeline.clock_column,
        window_seconds=window_seconds,
        sample_rate_hz=sample_rate_hz,
        overlap=overlap,
        amplitude="52 L-LTF imaginary/real pairs; HT20 indices 1:27+38:64; HT40 6:32+33:59",
        normalization="per-window global z-score; applied by training loader",
        receiver_mode="shared single-link backbone; each receiver is an example",
    )
    atomic_json(out / "preprocessing.json", summary)
    return summary
