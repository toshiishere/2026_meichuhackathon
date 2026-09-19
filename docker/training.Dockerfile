FROM python:3.12-slim-bookworm AS dependencies
ARG ROCM_GPU=gfx1152
ARG AMD_WHEEL_INDEX=https://stable.repo.amd.com/rocm/whl-next/
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 libgomp1 libnuma1 libdrm2 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
# These GPU-specific wheels match this machine's Radeon 840M/860M (gfx1152).
# Keep ROCm pinned when installing Ultralytics so pip cannot replace it with CUDA.
RUN pip install --no-cache-dir --index-url ${AMD_WHEEL_INDEX} --extra-index-url https://pypi.org/simple \
    "torch[device-${ROCM_GPU}]==2.13.0+rocm10.0.0" \
    "torchvision[device-${ROCM_GPU}]==0.28.0+rocm10.0.0" \
    "rocm[libraries,device-${ROCM_GPU}]==10.0.0"
COPY apps/common/requirements.txt apps/common/requirements.txt
COPY apps/training_service/requirements.txt apps/training_service/requirements.txt
RUN printf '%s\n' 'torch==2.13.0+rocm10.0.0' 'torchvision==0.28.0+rocm10.0.0' > /tmp/torch-constraints.txt \
    && pip install --no-cache-dir -c /tmp/torch-constraints.txt -r apps/training_service/requirements.txt \
    && pip check
# Build-time download pinned to the same CTR-GCN revision supplied in this workspace.
ARG CTR_GCN_COMMIT=e13d7582e281d06711eeecb380a472b278ae1663
RUN git clone https://github.com/Uason-Chen/CTR-GCN.git /opt/CTR-GCN \
    && git -C /opt/CTR-GCN checkout ${CTR_GCN_COMMIT} \
    && rm -rf /opt/CTR-GCN/.git
# MIOpen HIPRTC compiles normalization kernels at runtime and needs stdint.h.
# Without these headers the gfx1152 kernels fail with unresolved std::intmax_t.
RUN apt-get update && apt-get install -y --no-install-recommends libc6-dev \
    && rm -rf /var/lib/apt/lists/* && mkdir -p /tmp/ultralytics
ENV CTR_GCN_ROOT=/opt/CTR-GCN PYTHONUNBUFFERED=1 DATA_DIR=/data YOLO_CONFIG_DIR=/tmp/ultralytics
FROM dependencies AS runtime
COPY apps/common apps/common
COPY apps/training_service apps/training_service
COPY configs configs
COPY pose_labeling/inference.py pose_labeling/inference.py
COPY pose_labeling/ctrgcn_custom_4classes_best.pth pose_labeling/ctrgcn_custom_4classes_best.pth
COPY pose_labeling/yolov8n-pose.pt pose_labeling/yolov8n-pose.pt
COPY ["csi_model/Model Code/ESP_Fi_model.py", "csi_model/Model Code/ESP_Fi_model.py"]
COPY ["csi_model/Model Code/pretrained/ResNet18_full.pth", "csi_model/Model Code/pretrained/ResNet18_full.pth"]
COPY csi_model/finetune/session_tools/*.py csi_model/finetune/session_tools/
CMD ["uvicorn", "apps.training_service.app.main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
