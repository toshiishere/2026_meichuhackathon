from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator
from .config import DEFAULTS

ID = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$"
PORT = r"^(?:/dev/tty(?:USB|ACM)[0-9]+|/dev/serial/by-id/[A-Za-z0-9_.:+-]+|synthetic://(?:tx|rx[0-9]+))$"
TARGETS = Literal["esp32", "esp32s3", "esp32c3", "esp32c6"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Device(Strict):
    identity: str = Field(min_length=1, max_length=300)
    logical_name: str = Field(pattern=ID)
    role: Literal["csi_sender", "csi_receiver"]
    port: str = Field(pattern=PORT)
    target: TARGETS = "esp32c3"
    geometry: str = Field(default="", max_length=4000)
    firmware: dict = Field(default_factory=dict)


class Receiver(Strict):
    logical_name: str = Field(pattern=ID)
    port: str = Field(pattern=PORT)
    identity: str = Field(default="", max_length=300)
    geometry: str = Field(default="", max_length=4000)
    firmware: dict = Field(default_factory=dict)


class CameraConfig(Strict):
    device: str = Field(
        default="synthetic://camera",
        pattern=r"^(?:/dev/video[0-9]+|synthetic://camera)$",
    )
    fps: int = Field(default=DEFAULTS["camera_fps"], ge=1, le=120)
    width: int = Field(default=DEFAULTS["camera_width"], ge=160, le=3840)
    height: int = Field(default=DEFAULTS["camera_height"], ge=120, le=2160)
    geometry: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def even_dimensions(self):
        if self.width % 2 or self.height % 2:
            raise ValueError("H.264 recording requires even width and height")
        return self


class CollectionConfig(Strict):
    session_id: str = Field(pattern=ID)
    subject_id: str = Field(default="", max_length=120)
    room_id: str = Field(default="", max_length=120)
    layout_id: str = Field(default="", max_length=120)
    description: str = Field(default="", max_length=4000)
    activity_script: str = Field(default="", max_length=4000)
    notes: str = Field(default="", max_length=8000)
    layout_notes: str = Field(default="", max_length=4000)
    sender: Receiver
    receivers: list[Receiver] = Field(min_length=1, max_length=8)
    camera: CameraConfig
    baud_rate: int = Field(default=DEFAULTS["baud_rate"], ge=9600, le=3000000)
    expected_rate_hz: float = Field(default=DEFAULTS["expected_rate_hz"], ge=1, le=1000)
    min_rate_ratio: float = Field(default=DEFAULTS["min_rate_ratio"], gt=0, le=1)
    duration_seconds: float | None = Field(default=None, ge=1, le=86400)
    synthetic_loss: float = Field(default=DEFAULTS["synthetic_loss"], ge=0, le=0.5)
    synthetic_seed: int = Field(default=DEFAULTS["synthetic_seed"], ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def unique_devices(self):
        names = [x.logical_name for x in [self.sender, *self.receivers]]
        ports = [x.port for x in [self.sender, *self.receivers]]
        if len(set(names)) != len(names) or len(set(ports)) != len(ports):
            raise ValueError("Each sender/receiver needs a unique name and port")
        return self


class SerialRequest(Strict):
    port: str = Field(pattern=PORT)
    baud_rate: int = Field(default=DEFAULTS["baud_rate"], ge=9600, le=3000000)
    seconds: float = Field(default=5, ge=1, le=15)
    expected_rate_hz: float = Field(default=DEFAULTS["expected_rate_hz"], ge=1, le=1000)
    min_rate_ratio: float = Field(default=DEFAULTS["min_rate_ratio"], gt=0, le=1)


class FlashRequest(Strict):
    port: str = Field(pattern=PORT)
    target: TARGETS
    firmware: Literal["blink", "csi-send", "csi-recv"]
    operation: Literal["flash", "build", "rebuild", "clean-build"] = "flash"
    gpio: int | None = Field(default=None, ge=0, le=48)
    led_type: Literal["gpio", "rgb", "none"] = "gpio"
    active_low: bool = False
