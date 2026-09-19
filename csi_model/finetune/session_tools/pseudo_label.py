"""
Generate pseudo-labels (Static / Walking / Falling) for a collection session,
in the team's label format: start_time_ms,end_time_ms,movement (times in ms
from the start of the session video). This tool never emits Sitting or
Standing -- it can't tell them apart from Static.

Two independent signals, cross-checked:
1. Video motion detection: frame-to-frame grayscale difference at 2fps finds
   candidate "something moved" segments against a quiet baseline.
2. CSI amplitude shape (see csi_decode.py): within each motion segment,
   classify fall (motion energy sharply concentrated in a short burst, then
   quiet) vs walk (motion spread across the whole segment) -- the same
   signature validated against ESP-Fi HAR's own labeled `fall` samples.

Usage:
    python pseudo_label.py --session /path/to/session_dir --receiver mid

Writes pseudo_labels.csv into the session dir. This is a DRAFT labeling pass
-- spot-check a handful of segments against the video before trusting it.
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi_decode import decode_amplitude, load_csi_csv  # noqa: E402


MOVEMENT_FOR = {"walk": "Walking", "fall": "Falling"}


def get_video_duration(video_path):
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def compute_video_motion(video_path, frame_dir, fps=2, scale="160:120"):
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={fps},scale={scale}",
         "-q:v", "5", os.path.join(frame_dir, "f_%05d.jpg")],
        check=True, capture_output=True,
    )
    files = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
    grays = [np.array(Image.open(f).convert("L"), dtype=np.float32) for f in files]
    diffs = np.array([np.abs(grays[i] - grays[i - 1]).mean() for i in range(1, len(grays))])
    return diffs, 1.0 / fps


def detect_motion_segments(diffs, dt, bridge_gap_s=1.0, min_dur_s=1.0):
    med = np.median(diffs)
    mad = np.median(np.abs(diffs - med)) + 1e-6
    thresh = med + 3 * mad
    active = diffs > thresh

    segments = []
    i, n = 0, len(active)
    while i < n:
        if active[i]:
            start = i
            j = i
            while j < n:
                if active[j]:
                    j += 1
                elif j + 1 < n and active[j + 1] and (j - start) * dt < 100:
                    j += 1
                else:
                    break
            segments.append((start, j))
            i = j
        else:
            i += 1

    merged = []
    for s, e in segments:
        if merged and (s - merged[-1][1]) * dt <= bridge_gap_s / dt:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append([s, e])
    merged = [(s, e) for s, e in merged if (e - s) * dt >= min_dur_s]
    return merged


def video_time_to_host_ns(video_frames, t_s):
    return np.interp(t_s, video_frames["video_pts_s"], video_frames["host_timestamp_ns"])


def classify_segment(csi, lltf_idx, ns0, ns1, seg_dur):
    window = csi[(csi.host_timestamp_ns >= ns0) & (csi.host_timestamp_ns <= ns1)]
    if len(window) < 5:
        return "UNKNOWN (too few CSI packets)", None, len(window)

    amps = np.stack([decode_amplitude(d, lltf_idx) for d in window["data"].values])
    delta = np.abs(np.diff(amps, axis=0)).mean(axis=1)
    if delta.sum() == 0:
        return "UNKNOWN (flat CSI)", None, len(window)

    peak_idx = np.argmax(delta)
    ts = window["host_timestamp_ns"].values[1:] / 1e9
    peak_t = ts[peak_idx]
    near_mask = np.abs(ts - peak_t) <= 0.75
    concentration = float(delta[near_mask].sum() / delta.sum())

    label = "fall" if (concentration > 0.55 and seg_dur <= 6.0) else "walk"
    return label, concentration, len(window)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--receiver", default="mid", choices=["left", "mid", "right"])
    args = p.parse_args()

    session = args.session
    video_path = os.path.join(session, "raw", "video.mp4")
    video_frames_path = os.path.join(session, "raw", "video_frames.parquet")
    duration = get_video_duration(video_path)
    print(f"Session: {session}  video duration: {duration:.1f}s")

    with tempfile.TemporaryDirectory() as frame_dir:
        print("Extracting frames + computing video motion...")
        diffs, dt = compute_video_motion(video_path, frame_dir)
    segments = detect_motion_segments(diffs, dt)
    print(f"{len(segments)} candidate motion segments")

    # CSI: load the compressed csv if the plain one isn't there
    csv_path = os.path.join(session, "raw", f"csi_{args.receiver}.csv")
    zst_path = csv_path + ".zst"
    if not os.path.exists(csv_path) and os.path.exists(zst_path):
        print(f"Decompressing {zst_path}...")
        subprocess.run(["zstd", "-d", "-k", "-f", zst_path, "-o", csv_path], check=True, capture_output=True)

    csi, bw_name, lltf_idx = load_csi_csv(csv_path)
    print(f"Loaded {len(csi)} CSI rows from {args.receiver} receiver, bandwidth={bw_name}")

    video_frames = pd.read_parquet(video_frames_path) if video_frames_path.endswith(".parquet") else None
    import pyarrow.parquet as pq
    video_frames = pq.read_table(video_frames_path).to_pandas()

    results = []
    for s, e in segments:
        t0, t1 = (s + 1) * dt, (e + 1) * dt
        pad = 0.5
        ns0 = video_time_to_host_ns(video_frames, max(0, t0 - pad))
        ns1 = video_time_to_host_ns(video_frames, t1 + pad)
        label, conc, n = classify_segment(csi, lltf_idx, ns0, ns1, t1 - t0)
        results.append((t0, t1, label, conc, n))
        print(f"  {t0:6.1f}s - {t1:6.1f}s  {label:8s}  concentration={conc}  n_csi={n}")

    # Build the full timeline, filling gaps as Static. Segments the CSI couldn't
    # classify are left uncovered (and reported) rather than guessed at --
    # build_dataset.py only windows time that a label row covers.
    timeline = []
    prev_end = 0.0
    for t0, t1, label, conc, n in results:
        if t0 > prev_end + 0.05:
            timeline.append((prev_end, t0, "Static"))
        if label.startswith("UNKNOWN"):
            print(f"  WARNING: {t0:.1f}s - {t1:.1f}s left unlabeled ({label})")
        else:
            timeline.append((t0, t1, MOVEMENT_FOR[label]))
        prev_end = t1
    if prev_end < duration:
        timeline.append((prev_end, duration, "Static"))

    out_csv = os.path.join(session, "pseudo_labels.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_time_ms", "end_time_ms", "movement"])
        for t0, t1, movement in timeline:
            w.writerow([round(t0 * 1000), round(t1 * 1000), movement])

    counts = {m: sum(1 for r in timeline if r[2] == m) for m in ("Falling", "Walking", "Static")}
    print(f"\nWrote {out_csv}")
    print(", ".join(f"{n} {m}" for m, n in counts.items())
          + f" segments (bandwidth={bw_name}, receiver={args.receiver})")


if __name__ == "__main__":
    main()
