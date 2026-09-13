# Matches both existing generated sdkconfig headers: ESP-IDF 5.5.0.
FROM espressif/idf:v5.5
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends v4l-utils && rm -rf /var/lib/apt/lists/*
COPY apps/common/requirements.txt apps/common/requirements.txt
COPY apps/hardware_service/requirements.txt apps/hardware_service/requirements.txt
RUN /opt/esp/entrypoint.sh python -m pip install --no-cache-dir -r apps/hardware_service/requirements.txt
COPY apps/common apps/common
COPY apps/hardware_service apps/hardware_service
COPY configs configs
COPY firmware firmware
COPY scripts scripts
ENV PYTHONUNBUFFERED=1 DATA_DIR=/data HARDWARE_MODE=real
CMD ["uvicorn", "apps.hardware_service.app.main:app", "--host", "0.0.0.0", "--port", "8001", "--ws-max-size", "2097152", "--ws-per-message-deflate", "false"]
