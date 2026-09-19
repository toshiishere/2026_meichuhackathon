import cv2
import torch
import numpy as np
import os
import sys
from collections import deque
from ultralytics import YOLO

# Ensure CTR-GCN repo module is accessible
REPO_URL = "https://github.com/Uason-Chen/CTR-GCN.git"
REPO_DIR = "CTR-GCN"

if not os.path.exists(REPO_DIR):
    print("--- Automatically cloning CTR-GCN repository ---")
    os.system(f"git clone {REPO_URL}")

sys.path.append(os.path.abspath(REPO_DIR))

from model.ctrgcn import Model as CTRGCN_Model

# Target action labels mapped to class indices
ACTION_LABELS = {
    0: "Sitting",
    1: "Standing",
    2: "Walking",
    3: "Falling"
}

WINDOW_SIZE = 64

def yolo_to_ntu25(yolo_kpts):
    """
    Maps YOLO 17-keypoint COCO format to NTU 25-keypoint format.
    """
    ntu_kpts = np.zeros((25, 3), dtype=np.float32)
    mapping = {
        3: 0, 4: 5, 5: 7, 6: 9, 8: 6, 9: 8, 10: 10, 
        12: 11, 13: 13, 14: 15, 16: 12, 17: 14, 18: 16
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
        ntu_kpts[1, 2] = (ntu_kpts[2, 2] + (yolo_kpts[11, 2] + yolo_kpts[12, 2])/2) / 2
        
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

def run_text_inference(video_path, output_csv="action_results.csv", weight_path="ctrgcn_custom_4classes_best.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initializing inference pipeline (Device: {device})")
    
    if not os.path.exists(video_path):
        print(f"❌ Input video file not found: {video_path}")
        return
        
    # 1. Load YOLO model
    print("--- Loading YOLOv8-Pose Model ---")
    yolo_model = YOLO("yolov8n-pose.pt")
    
    # 2. Load fine-tuned CTR-GCN model
    print(f"--- Loading Fine-Tuned Weights: {weight_path} ---")
    action_model = CTRGCN_Model(
        num_class=4, 
        num_point=25, 
        num_person=2, 
        graph='graph.ntu_rgb_d.Graph', 
        graph_args={'labeling_mode': 'spatial'},
        in_channels=3,
        drop_out=0.0
    ).to(device)
    
    if os.path.exists(weight_path):
        action_model.load_state_dict(torch.load(weight_path, map_location=device))
        print("✅ Weights loaded successfully!")
    else:
        print("❌ Weight file not found. Terminating execution.")
        return
        
    action_model.eval()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    target_buffer_size = max(16, int(fps * (64.0 / 30.0)))
    print(f"⏱️ Video FPS: {fps:.2f} (Height: {video_height}). Dynamic buffer size: {target_buffer_size} frames.")
    
    pose_buffer = deque(maxlen=target_buffer_size)
    last_logged_action = None
    frame_count = 0
    
    # Anti-flicker variables
    frames_to_confirm = int(fps * 0.8)
    candidate_action = None
    candidate_count = 0
    
    # Continuous action state tracking
    active_continuous = None
    active_continuous_start = 0
    
    with open(output_csv, "w", encoding="utf-8") as subtitle_file:
        subtitle_file.write("start_time,end_time,label\n")
        print(f"🎬 Analyzing video: {video_path} (FPS: {fps:.2f})")
        print("--- Action Log ---")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            
            results = yolo_model(frame, verbose=False)
            keypoints_tensor = results[0].keypoints.data
            
            current_prediction = None

            if len(keypoints_tensor) > 0:
                person_kpts = keypoints_tensor[0].cpu().numpy()
                ntu25_kpts = yolo_to_ntu25(person_kpts)
                pose_buffer.append(ntu25_kpts)
                
                if len(pose_buffer) == target_buffer_size:
                    is_moving = is_skeleton_moving(pose_buffer, video_height, num_frames=16, threshold=0.03)
                    
                    if not is_moving:
                        current_prediction = "Static"
                    else:
                        resampled_seq = resample_skeleton_sequence(pose_buffer, WINDOW_SIZE)
                        norm_seq = normalize_sequence(resampled_seq, video_height)
                        input_tensor = format_buffer_to_ctrgcn(norm_seq).to(device)
                        
                        with torch.no_grad():
                            outputs = action_model(input_tensor)
                            _, predicted_idx = torch.max(outputs.data, 1)
                            current_prediction = ACTION_LABELS.get(predicted_idx.item(), "Unknown")
            else:
                # No skeleton detected, treat as Static and clear buffer
                current_prediction = "Static"
                pose_buffer.clear()
                            
            if current_prediction is not None:
                # Anti-flicker & candidate filtering logic
                if current_prediction == candidate_action:
                    candidate_count += 1
                else:
                    candidate_action = current_prediction
                    candidate_count = 1
                    
                # Continuous actions require 2.0s confirmation, instant actions require 0.8s
                required_frames = int(fps * 2.0) if candidate_action == "Static" else frames_to_confirm
                if candidate_count >= required_frames and candidate_action != last_logged_action:
                    start_frame = max(0, frame_count - required_frames + 1)
                    t_ms = int((start_frame / fps) * 1000)
                    
                    is_continuous = candidate_action in ["Walking", "Static"]
                    
                    if is_continuous:
                        if active_continuous is not None:
                            subtitle_file.write(f"{active_continuous_start},{t_ms},{active_continuous}\n")
                        active_continuous = candidate_action
                        active_continuous_start = t_ms
                    else:
                        if active_continuous is not None:
                            end_t = max(0, t_ms - 1000)
                            if end_t > active_continuous_start:
                                subtitle_file.write(f"{active_continuous_start},{end_t},{active_continuous}\n")
                            active_continuous = None
                            
                        start_t = max(0, t_ms - 1000)
                        end_t = t_ms + 1000
                        subtitle_file.write(f"{start_t},{end_t},{candidate_action}\n")
                    
                    print(f"[Time: {t_ms} ms] Transition -> {candidate_action}")
                    last_logged_action = candidate_action

        # Write remaining active continuous state if present
        if active_continuous is not None:
            final_t_ms = int((frame_count / fps) * 1000)
            subtitle_file.write(f"{active_continuous_start},{final_t_ms},{active_continuous}\n")

    cap.release()
    print("\n✅ Video analysis complete!")
    print(f"💾 Output saved to CSV: {output_csv}")

if __name__ == "__main__":
    run_text_inference("test.mp4", "action_results.csv")