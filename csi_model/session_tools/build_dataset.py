"""
Build an ESP-Fi-HAR-shaped training dataset from one or more collection
sessions, using a configurable sliding window over each labeled interval.

Output layout matches what Model Code/dataset.py and loso_data.py expect --
one .mat file per window, holding a `CSIamp` array of shape (T, 52), sorted
into one folder per class:

    <out>/<label>/<session_id>-<receiver>-w<window_idx>.mat

T = round(window_seconds * sample_rate_hz) is NOT fixed at the paper's 950
(10s @ ~100Hz) -- it follows whatever --window-seconds you pass (default 2s
-> T=200 @ 100Hz). Model Code/dataset.py and loso_data.py were hardcoded to
only accept T=950; both were patched to accept any T (still validating the
52-subcarrier dimension) so this pipeline's output actually loads.

Label format (one file per session, CSV):

    start_time_ms,end_time_ms,movement

Times are ms from the start of the session video. `movement` must be one of
Static, Walking, Sitting, Standing, Falling; anything else is an error. Each
movement gets its own output folder. Time not covered by any row is simply
not windowed.

Windowing policy:
- Interval >= window: slide with stride = window * (1 - overlap), starting
  at the interval's start; if the last full-stride window doesn't reach the
  interval's end, one extra window is placed flush against the end so the
  tail isn't lost (it can overlap the previous window more than the
  configured overlap -- that's fine, it's still fully inside the interval).
- Interval < window (this matters: some real `fall` events are shorter than
  a 2s window): ONE window is emitted, centered on the interval's midpoint
  and clipped to the session's time range. It will include a bit of
  surrounding context outside the labeled interval -- this mirrors how the
  paper's own `fall` samples look (a brief dynamic burst plus a static
  aftermath inside one 10s trial), not a tight crop of only the label.

Each window is resampled (per-subcarrier linear interpolation over
host_timestamp_ns) onto a uniform T-point time grid, independently per
receiver -- this keeps every receiver's window for "the same" time span
exactly aligned in shape and timing, which matters for later multi-link use
(ESP_Fi_ResNet18_MultiLink expects simultaneous, aligned per-link windows).

Usage:
    python build_dataset.py --session /path/to/session_dir [/path/to/other_session ...] \
        --out ./built_dataset --window-seconds 2.0 --overlap 0.5

    # narrower/wider window, single receiver only:
    python build_dataset.py --session ./session_X --out ./built_dataset \
        --window-seconds 1.0 --overlap 0.5 --receivers mid
"""
import argparse
import csv
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi_decode import decode_amplitude, load_csi_csv  # noqa: E402


def ensure_decompressed(session, receiver):
    csv_path = os.path.join(session, "raw", f"csi_{receiver}.csv")
    zst_path = csv_path + ".zst"
    if not os.path.exists(csv_path) and os.path.exists(zst_path):
        subprocess.run(["zstd", "-d", "-k", "-f", zst_path, "-o", csv_path],
                        check=True, capture_output=True)
    return csv_path


MOVEMENTS = ["Static", "Walking", "Sitting", "Standing", "Falling"]
LABEL_COLUMNS = ["start_time_ms", "end_time_ms", "movement"]


def load_labels(path):
    """Read a label file with columns start_time_ms,end_time_ms,movement
    (times in ms from the start of the session video). Returns a DataFrame
    with start_s/end_s/label columns. Raises on malformed rows rather than
    silently building windows -- or a stray class folder -- from a typo.
    """
    df = pd.read_csv(path)
    missing = [c for c in LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; expected {LABEL_COLUMNS}")
    bad = sorted(set(df["movement"].astype(str)) - set(MOVEMENTS))
    if bad:
        raise ValueError(f"{path}: unknown movement(s) {bad}; valid: {MOVEMENTS}")
    if (df["end_time_ms"] <= df["start_time_ms"]).any():
        raise ValueError(f"{path}: found rows with end_time_ms <= start_time_ms")
    return pd.DataFrame({
        "start_s": df["start_time_ms"] / 1000.0,
        "end_s": df["end_time_ms"] / 1000.0,
        "label": df["movement"],
    })


def video_time_to_host_ns(video_frames, t_s):
    return float(np.interp(t_s, video_frames["video_pts_s"], video_frames["host_timestamp_ns"]))


def make_windows(start_s, end_s, window_s, stride_s):
    """Returns a list of (win_start_s, win_end_s) covering [start_s, end_s)."""
    dur = end_s - start_s
    if dur <= window_s:
        mid = (start_s + end_s) / 2.0
        return [(mid - window_s / 2.0, mid + window_s / 2.0)]

    windows = []
    t = start_s
    while t + window_s <= end_s:
        windows.append((t, t + window_s))
        t += stride_s
    if not windows or windows[-1][1] < end_s:
        windows.append((end_s - window_s, end_s))
    return windows


def resample_window(csi, lltf_idx, ns0, ns1, n_samples):
    """Pull packets in [ns0, ns1), decode amplitude, resample to n_samples
    points on a uniform time grid via per-subcarrier linear interpolation.
    Returns None if too few packets are available to interpolate meaningfully.
    """
    window = csi[(csi.host_timestamp_ns >= ns0) & (csi.host_timestamp_ns < ns1)]
    if len(window) < 3:
        return None

    ts = window["host_timestamp_ns"].values.astype(np.float64)
    amps = np.stack([decode_amplitude(d, lltf_idx) for d in window["data"].values])  # (n_pkts, 52)

    grid = np.linspace(ns0, ns1, n_samples, endpoint=False)
    out = np.empty((n_samples, amps.shape[1]), dtype=np.float32)
    for c in range(amps.shape[1]):
        out[:, c] = np.interp(grid, ts, amps[:, c])
    return out


def process_session(session, out_dir, window_s, overlap, sample_rate_hz, receivers,
                     labels_file, manifest_rows):
    session_id = os.path.basename(os.path.normpath(session))
    labels_path = os.path.join(session, labels_file)
    if not os.path.exists(labels_path):
        print(f"  SKIP {session_id}: no {labels_file}")
        return

    labels = load_labels(labels_path)

    video_frames_path = os.path.join(session, "raw", "video_frames.parquet")
    video_frames = pq.read_table(video_frames_path).to_pandas()

    stride_s = window_s * (1.0 - overlap)
    n_samples = round(window_s * sample_rate_hz)

    csi_by_receiver = {}
    for r in receivers:
        csv_path = ensure_decompressed(session, r)
        csi, bw_name, lltf_idx = load_csi_csv(csv_path)
        csi_by_receiver[r] = (csi, bw_name, lltf_idx)
        print(f"  {session_id}/{r}: {len(csi)} packets, bandwidth={bw_name}")

    window_counter = 0
    written = 0
    for interval_idx, (_, row) in enumerate(labels.iterrows()):
        label = row["label"]
        interval_id = f"{session_id}-interval{interval_idx:04d}"
        wins = make_windows(row["start_s"], row["end_s"], window_s, stride_s)

        for w_start, w_end in wins:
            ns0 = video_time_to_host_ns(video_frames, max(0, w_start))
            ns1 = video_time_to_host_ns(video_frames, w_end)

            for r in receivers:
                csi, bw_name, lltf_idx = csi_by_receiver[r]
                amp = resample_window(csi, lltf_idx, ns0, ns1, n_samples)
                if amp is None:
                    continue

                class_dir = os.path.join(out_dir, label)
                os.makedirs(class_dir, exist_ok=True)
                fname = f"{session_id}-{r}-w{window_counter:05d}.mat"
                fpath = os.path.join(class_dir, fname)
                sio.savemat(fpath, {"CSIamp": amp})
                written += 1

                # interval_id/interval_start_s/interval_end_s identify the ORIGINAL
                # labeled interval a window came from (before sliding-window
                # splitting) -- required for leak-free train/val splitting, since
                # overlapping windows from the same interval are near-duplicates.
                manifest_rows.append([
                    session_id, r, label, round(w_start, 3), round(w_end, 3),
                    n_samples, bw_name, fname,
                    interval_id, round(row["start_s"], 3), round(row["end_s"], 3),
                ])
            window_counter += 1

    print(f"  {session_id}: {window_counter} label-windows -> {written} .mat files written")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", nargs="+", required=True, help="one or more session directories")
    p.add_argument("--out", required=True)
    p.add_argument("--window-seconds", type=float, default=2.0)
    p.add_argument("--overlap", type=float, default=0.5, help="fraction, 0-1")
    p.add_argument("--sample-rate-hz", type=float, default=100.0,
                    help="target samples/sec after resampling; T = round(window_seconds * this)")
    p.add_argument("--receivers", nargs="+", default=["left", "mid", "right"])
    p.add_argument("--labels-file", default="pseudo_labels.csv",
                    help="label CSV (start_time_ms,end_time_ms,movement), looked up "
                         "inside each session dir, or an absolute path")
    args = p.parse_args()

    if not (0.0 <= args.overlap < 1.0):
        raise ValueError("--overlap must be in [0, 1)")

    os.makedirs(args.out, exist_ok=True)
    n_samples = round(args.window_seconds * args.sample_rate_hz)
    print(f"window={args.window_seconds}s overlap={args.overlap} -> "
          f"stride={args.window_seconds * (1 - args.overlap):.2f}s, T={n_samples} samples/window")

    manifest_rows = []
    for session in args.session:
        process_session(session, args.out, args.window_seconds, args.overlap,
                         args.sample_rate_hz, args.receivers, args.labels_file, manifest_rows)

    manifest_path = os.path.join(args.out, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session_id", "receiver", "label", "window_start_s", "window_end_s",
                    "n_samples", "bandwidth", "filename",
                    "interval_id", "interval_start_s", "interval_end_s"])
        w.writerows(manifest_rows)
    print(f"\nWrote {len(manifest_rows)} rows to {manifest_path}")

    counts = {}
    for row in manifest_rows:
        counts[row[2]] = counts.get(row[2], 0) + 1
    print("Class counts:", counts)


if __name__ == "__main__":
    main()
