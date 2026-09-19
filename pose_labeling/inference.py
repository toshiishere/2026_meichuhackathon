import cv2
import torch
import numpy as np
import os
import sys
from collections import deque
import argparse

from pathlib import Path
import argparse
import csv
import json
import av

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apps.training_service.app.timeline import Timeline

# Target action labels mapped to class indices
ACTION_LABELS = {0: "Sitting", 1: "Standing", 2: "Walking", 3: "Falling"}

WINDOW_SIZE = 64


def yolo_to_ntu25(yolo_kpts):
    """
    Maps YOLO 17-keypoint COCO format to NTU 25-keypoint format.
    """
    ntu_kpts = np.zeros((25, 3), dtype=np.float32)
    mapping = {
        3: 0,
        4: 5,
        5: 7,
        6: 9,
        8: 6,
        9: 8,
        10: 10,
        12: 11,
        13: 13,
        14: 15,
        16: 12,
        17: 14,
        18: 16,
    }

    for ntu_idx, coco_idx in mapping.items():
        if coco_idx < len(yolo_kpts):
            ntu_kpts[ntu_idx] = yolo_kpts[coco_idx]

    # Interpolate neck and spine center
    if len(yolo_kpts) > 12:
        ntu_kpts[2, :2] = (yolo_kpts[5, :2] + yolo_kpts[6, :2]) / 2
        ntu_kpts[2, 2] = (yolo_kpts[5, 2] + yolo_kpts[6, 2]) / 2

        hip_center = (yolo_kpts[11, :2] + yolo_kpts[12, :2]) / 2
        ntu_kpts[1, :2] = (ntu_kpts[2, :2] + hip_center) / 2
        ntu_kpts[1, 2] = (
            ntu_kpts[2, 2] + (yolo_kpts[11, 2] + yolo_kpts[12, 2]) / 2
        ) / 2

    return ntu_kpts


def normalize_sequence(sequence, video_height):
    """
    Sequence-level normalization: Origin at mid-hip of 1st frame, scaled by video height.
    """
    norm_seq = sequence.copy()
    first_frame = norm_seq[0]

    if first_frame[12, 2] > 0 and first_frame[16, 2] > 0:
        root_x = (first_frame[12, 0] + first_frame[16, 0]) / 2.0
        root_y = (first_frame[12, 1] + first_frame[16, 1]) / 2.0
    else:
        valid_idx = first_frame[:, 2] > 0
        if np.sum(valid_idx) > 0:
            root_x = np.mean(first_frame[valid_idx, 0])
            root_y = np.mean(first_frame[valid_idx, 1])
        else:
            root_x, root_y = 0.0, 0.0

    scale = float(video_height) if video_height > 0 else 1080.0
    final_seq = np.zeros((len(norm_seq), 25, 3), dtype=np.float32)

    for t in range(len(norm_seq)):
        for i in range(25):
            if norm_seq[t, i, 2] > 0:
                final_seq[t, i, 0] = (norm_seq[t, i, 0] - root_x) / scale
                final_seq[t, i, 1] = (norm_seq[t, i, 1] - root_y) / scale
                final_seq[t, i, 2] = 1.0  # Binary confidence

    return final_seq


def resample_skeleton_sequence(buffer, target_len=64):
    """
    Resamples variable length skeleton buffer to 64 frames.
    """
    buffer_array = np.array(buffer)
    indices = np.linspace(0, len(buffer_array) - 1, target_len).astype(int)
    return buffer_array[indices]


def format_buffer_to_ctrgcn(sequence_array):
    """
    Formats 64-frame sequence to CTR-GCN tensor format (1, 3, 64, 25, 2).
    """
    input_tensor = np.zeros((1, 3, WINDOW_SIZE, 25, 2), dtype=np.float32)
    for t in range(WINDOW_SIZE):
        kpts = sequence_array[t]
        input_tensor[0, 0, t, :, 0] = kpts[:, 0]  # X
        input_tensor[0, 1, t, :, 0] = kpts[:, 1]  # Y
        input_tensor[0, 2, t, :, 0] = kpts[:, 2]  # Confidence
    return torch.tensor(input_tensor, dtype=torch.float32)


def is_skeleton_moving(buffer, video_height, num_frames=16, threshold=0.03):
    """
    Motion Gate: Checks standard deviation of stable torso keypoints over the recent N frames.
    Returns True if moving, False if static.
    """
    if len(buffer) < num_frames:
        return True

    recent_frames = list(buffer)[-num_frames:]
    recent_seq = np.array(recent_frames)

    stable_joints = [0, 1, 2, 4, 8, 12, 16]  # Spine, neck, shoulders, hips
    coords = recent_seq[:, stable_joints, :2]
    valid_mask = recent_seq[:, stable_joints, 2] > 0

    max_std = 0.0
    for j in range(len(stable_joints)):
        valid_coords = coords[valid_mask[:, j], j, :]
        if len(valid_coords) > 1:
            std_x = np.std(valid_coords[:, 0])
            std_y = np.std(valid_coords[:, 1])
            max_std = max(max_std, std_x, std_y)

    norm_std = max_std / float(video_height if video_height > 0 else 1080.0)
    return norm_std > threshold


def run_text_inference(
    video_path, output_csv=None, weight_path=None, frame_index=None, device="cuda:0"
):
    """Label a collector video using its actual PTS, never frame_count / FPS.

    Missing detections or >500ms camera gaps break the action history. Labels
    are automatic predictions for a single visible person, not reviewed truth.
    On ROCm, PyTorch and Ultralytics intentionally use the cuda device API.
    """
    from ultralytics import YOLO

    repo = Path(os.environ.get("CTR_GCN_ROOT", Path(__file__).parent / "CTR-GCN"))
    if not repo.is_dir():
        raise FileNotFoundError(
            "CTR-GCN source is missing; build the training container"
        )
    sys.path.insert(0, str(repo))
    from model.ctrgcn import Model as CTRGCN_Model

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError(
            "ROCm GPU unavailable. Expose /dev/kfd and /dev/dri to the training container; CPU fallback is disabled."
        )
    video_path = Path(video_path)
    output_csv = Path(
        output_csv or video_path.parent.parent / "train/action_results.csv"
    )
    weight_path = Path(
        weight_path or Path(__file__).parent / "ctrgcn_custom_4classes_best.pth"
    )
    yolo_path = Path(__file__).parent / "yolov8n-pose.pt"
    for path in (video_path, weight_path, yolo_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    timeline = Timeline(frame_index or video_path.parent / "video_frames.parquet")
    print(
        f"ROCm {torch.version.hip}; GPU: {torch.cuda.get_device_name(device)}",
        flush=True,
    )
    yolo_model = YOLO(str(yolo_path)).to(device)
    action_model = CTRGCN_Model(
        num_class=4,
        num_point=25,
        num_person=2,
        graph="graph.ntu_rgb_d.Graph",
        graph_args={"labeling_mode": "spatial"},
        in_channels=3,
        drop_out=0.0,
    ).to(device)
    action_model.load_state_dict(
        torch.load(weight_path, map_location=device, weights_only=True)
    )
    action_model.eval()
    history = deque()
    segments = []
    active = candidate = None
    active_start = candidate_start = 0.0
    previous = None
    count = 0

    def close(end):
        nonlocal active
        if active is not None and end > active_start:
            segments.append((active_start, end, active))
        active = None

    with av.open(str(video_path)) as video, torch.inference_mode():
        for idx, decoded in enumerate(video.decode(video=0)):
            if idx >= len(timeline.pts) or decoded.pts is None:
                raise ValueError(
                    "Video and frame index differ; cannot align labels with CSI"
                )
            stamp = float(decoded.pts * decoded.time_base)
            if abs(stamp - timeline.pts[idx]) > 0.002:
                raise ValueError(f"Video PTS differs from frame index at frame {idx}")
            if previous is not None and stamp - previous > 0.5:
                close(previous)
                history.clear()
                candidate = None
            frame = decoded.to_ndarray(format="bgr24")
            results = yolo_model.predict(frame, device=device, verbose=False, max_det=1)
            keypoints = results[0].keypoints
            if keypoints is None or len(keypoints.data) == 0:
                close(previous if previous is not None else stamp)
                history.clear()
                candidate = None
            else:
                history.append((stamp, yolo_to_ntu25(keypoints.data[0].cpu().numpy())))
                span = WINDOW_SIZE / 30.0
                while len(history) > 2 and history[1][0] < stamp - span:
                    history.popleft()
                if stamp - history[0][0] >= span - 0.1:
                    # Resample over elapsed video time, including VFR capture.
                    times = np.array([item[0] for item in history])
                    indices = np.searchsorted(
                        times, np.linspace(times[0], stamp, WINDOW_SIZE)
                    )
                    skeleton = np.array([item[1] for item in history])[indices]
                    if not is_skeleton_moving(skeleton, frame.shape[0]):
                        prediction = "Static"
                    else:
                        tensor = format_buffer_to_ctrgcn(
                            normalize_sequence(skeleton, frame.shape[0])
                        ).to(device)
                        prediction = ACTION_LABELS[
                            int(action_model(tensor).argmax(dim=1).item())
                        ]
                    if candidate != prediction:
                        candidate, candidate_start = prediction, stamp
                    confirmation = 2.0 if candidate == "Static" else 0.8
                    if stamp - candidate_start >= confirmation and candidate != active:
                        close(candidate_start)
                        active, active_start = candidate, candidate_start
                        print(f"Label {candidate_start:.3f}s: {active}", flush=True)
            previous = stamp
            count += 1
            if count % max(1, len(timeline.pts) // 100) == 0:
                print(f"Labeling {count}/{len(timeline.pts)} frames", flush=True)
    if count != len(timeline.pts):
        raise ValueError("Video ended before the recorded frame index")
    close(float(timeline.pts[-1]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_name("." + output_csv.name + ".tmp")
    with temporary.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "start_time",
                "end_time",
                "label",
                "start_host_timestamp_ns",
                "end_host_timestamp_ns",
            ]
        )
        for start, end, label in segments:
            writer.writerow(
                [
                    f"{start * 1000:.6f}",
                    f"{end * 1000:.6f}",
                    label,
                    timeline.host_ns(start),
                    timeline.host_ns(end),
                ]
            )
    os.replace(temporary, output_csv)
    output_csv.with_suffix(".json").write_text(
        json.dumps(
            dict(
                source="YOLOv8 pose + CTR-GCN + motion gate; automatically generated",
                video=str(video_path),
                frames=count,
                intervals=len(segments),
                time_unit="milliseconds",
                time_origin="video PTS",
                host_clock_column=timeline.clock_column,
                gpu=torch.cuda.get_device_name(device),
                rocm=torch.version.hip,
            ),
            indent=2,
        )
    )
    print(f"Wrote {len(segments)} intervals to {output_csv}", flush=True)
    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label one recorded session on ROCm")
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_text_inference(
        args.session / "raw/video.mp4",
        args.output or args.session / "train/action_results.csv",
        frame_index=args.session / "raw/video_frames.parquet",
    )
