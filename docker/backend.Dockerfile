FROM python:3.11.12-slim-bookworm
WORKDIR /workspace
COPY apps/common/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY apps/common apps/common
COPY apps/backend apps/backend
COPY configs configs
ENV PYTHONUNBUFFERED=1 DATA_DIR=/data
CMD ["uvicorn", "apps.backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
