FROM python:3.11.12-slim-bookworm
WORKDIR /app
COPY dc_bot/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY dc_bot .
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "bot:app", "--host", "0.0.0.0", "--port", "8003"]
