"""Internal HTTP adapter for the importable Ryzen AI runtime."""

from contextlib import asynccontextmanager
import io
import json
import re
import threading

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from apps.common.config import DATA
from apps.common.schemas import ID
from .runtime import NpuModel, digest

MODEL_PATH = re.compile(r"train/runs/[a-f0-9]{32}/finetuned_resnet18\.pth")


class PrepareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_session_id: str = Field(pattern=ID)


def model_artifacts(session_id: str):
    session = DATA / "sessions" / session_id
    if not session.is_dir() or session.is_symlink():
        raise ValueError("Model session not found")
    pointer = session / "train/model.json"
    if not pointer.is_file() or pointer.is_symlink():
        raise ValueError("Model metadata not found")
    metadata = json.loads(pointer.read_text())
    relative = metadata.get("model_path", "")
    if metadata.get("architecture") != "ESP_Fi_ResNet18" or not MODEL_PATH.fullmatch(
        relative
    ):
        raise ValueError("Unsupported or unsafe model metadata")
    checkpoint = session / relative
    classes_file = checkpoint.parent / "classes.json"
    for path in (checkpoint, classes_file):
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(session.resolve())
        ):
            raise ValueError("Missing or unsafe model artifact")
    if digest(checkpoint) != metadata.get("model_sha256"):
        raise ValueError("Model checksum mismatch")
    class_data = json.loads(classes_file.read_text())
    classes = class_data.get("class_names", [])
    if (
        len(classes) < 2
        or len(set(classes)) != len(classes)
        or class_data.get("class_to_idx")
        != {name: index for index, name in enumerate(classes)}
    ):
        raise ValueError("Invalid model classes")
    return metadata, checkpoint, classes


class Models:
    def __init__(self):
        self.lock = threading.Lock()
        self.key = None
        self.model = None

    def prepare(self, session_id):
        metadata, checkpoint, classes = model_artifacts(session_id)
        key = metadata["model_sha256"]
        with self.lock:
            if self.key != key:
                self.model = None
                self.model = NpuModel(metadata, checkpoint, classes)
                self.key = key
        return {
            "model_key": key,
            "classes": classes,
            "provider": "VitisAIExecutionProvider",
            "quantization": "XINT8",
        }

    def infer(self, key, batch):
        with self.lock:
            if self.key != key or self.model is None:
                raise KeyError("NPU model is not prepared")
            return self.model.infer(batch)

    def release(self, key):
        with self.lock:
            if self.key == key:
                self.model = None
                self.key = None


models = Models()


@asynccontextmanager
async def lifespan(app):
    yield
    if models.key:
        models.release(models.key)


app = FastAPI(title="CSI Ryzen AI worker", lifespan=lifespan)


@app.get("/health")
def health():
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
    except Exception as error:
        raise HTTPException(503, f"Ryzen AI runtime unavailable: {error}") from error
    if "VitisAIExecutionProvider" not in providers:
        raise HTTPException(503, "VitisAIExecutionProvider unavailable")
    return {"status": "ok", "providers": providers}


@app.post("/models/prepare")
def prepare(body: PrepareRequest):
    try:
        return models.prepare(body.model_session_id)
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        raise HTTPException(400, str(error)) from error


@app.post("/models/{key}/infer")
async def infer(key: str, request: Request):
    if not re.fullmatch(r"[a-f0-9]{64}", key):
        raise HTTPException(404, "NPU model not found")
    try:
        batch = np.load(io.BytesIO(await request.body()), allow_pickle=False)
        scores, elapsed = models.infer(key, batch)
        return {"scores": scores.tolist(), "inference_ms": elapsed}
    except KeyError as error:
        raise HTTPException(404, str(error)) from error
    except (ValueError, OSError) as error:
        raise HTTPException(400, str(error)) from error


@app.post("/models/{key}/release")
def release(key: str):
    models.release(key)
    return {"released": True}
