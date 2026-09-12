FROM python:3.11.12-slim-bookworm
WORKDIR /workspace
COPY apps/common/requirements.txt apps/common/requirements.txt
COPY apps/hardware_service/requirements.txt apps/hardware_service/requirements.txt
RUN pip install --no-cache-dir -r apps/hardware_service/requirements.txt
COPY apps apps
COPY configs configs
COPY firmware firmware
COPY scripts scripts
COPY tests tests
COPY pytest.ini pytest.ini
RUN pip install --no-cache-dir -r tests/requirements.txt
ENV PYTHONUNBUFFERED=1 DATA_DIR=/data HARDWARE_MODE=synthetic
CMD ["uvicorn", "apps.hardware_service.app.main:app", "--host", "0.0.0.0", "--port", "8001"]
