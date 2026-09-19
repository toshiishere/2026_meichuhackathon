import torch
import time
import numpy as np
import os
import platform
import argparse

# =========================
# 1. Parse Arguments
# =========================
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='CNN')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to model checkpoint (.pth)')
parser.add_argument('--runs', type=int, default=200)
args = parser.parse_args()

# =========================
# 2. Environment Information
# =========================
print("===== Environment Info =====")
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"CPU: {platform.processor()}")
print(f"Num CPU threads: {torch.get_num_threads()}")
print("============================\n")

device = torch.device("cpu")
torch.set_num_threads(os.cpu_count())

# =========================
# 3. Build Model
# =========================
from ESP_Fi_model import CNN

num_classes = 7

if args.model == "CNN":
    model = CNN(num_classes=num_classes)
else:
    raise ValueError("Currently only CNN is supported in latency test.")

model.to(device)
model.eval()

# =========================
# 4. Load Checkpoint (Optional)
# =========================
if args.checkpoint and os.path.exists(args.checkpoint):
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Checkpoint loaded from {args.checkpoint}")
else:
    print("No checkpoint provided. Using random weights.")

# =========================
# 5. Model Size Statistics
# =========================
num_params = sum(p.numel() for p in model.parameters())
model_size_mb = num_params * 4 / (1024 ** 2)

print("\n===== Model Size =====")
print(f"Total parameters : {num_params:,}")
print(f"Model size (FP32): {model_size_mb:.2f} MB")

# =========================
# 6. Prepare Input Tensor
# =========================
input_shape = (1, 1, 950, 52)
input_tensor = torch.randn(*input_shape, device=device)

# =========================
# 7. Warm-up
# =========================
print("\nWarming up model...")
with torch.no_grad():
    for _ in range(20):
        _ = model(input_tensor)
print("Warm-up completed.")

# =========================
# 8. Inference Latency
# =========================
latencies = []

print(f"\nMeasuring inference latency over {args.runs} runs...")
with torch.no_grad():
    for _ in range(args.runs):
        start_time = time.perf_counter()
        _ = model(input_tensor)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)

latencies = np.array(latencies)

# =========================
# 9. Report
# =========================
print("\n===== CPU Inference Latency =====")
print(f"Input size  : {' × '.join(map(str, input_shape))}")
print(f"Batch size  : {input_shape[0]}")
print(f"Runs        : {args.runs}")
print(f"Avg latency : {latencies.mean():.2f} ms")
print(f"Std latency : {latencies.std():.2f} ms")
print(f"Min latency : {latencies.min():.2f} ms")
print(f"Max latency : {latencies.max():.2f} ms")