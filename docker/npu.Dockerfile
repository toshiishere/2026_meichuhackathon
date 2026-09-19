FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 libboost-filesystem1.83.0 libexpat1 libglib2.0-0 libgomp1 \
    libnuma1 libdrm2 libncurses6 pciutils ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY apps/common apps/common
COPY apps/npu_service apps/npu_service
COPY configs configs
COPY ["csi_model/Model Code/ESP_Fi_model.py", "csi_model/Model Code/ESP_Fi_model.py"]
ENV PYTHONUNBUFFERED=1 DATA_DIR=/data RYZEN_AI_INSTALLATION_PATH=/opt/ryzen-ai \
    XILINX_XRT=/opt/xilinx/xrt \
    LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:/opt/ryzen-ai/lib/python3.12/site-packages/flexml/flexml_extras/lib:/opt/ryzen-ai/lib/python3.12/site-packages/onnxruntime/capi:/opt/ryzen-ai/lib/python3.12/site-packages/voe/lib
CMD ["/opt/ryzen-ai/bin/python", "-m", "uvicorn", "apps.npu_service.app.main:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "1"]
