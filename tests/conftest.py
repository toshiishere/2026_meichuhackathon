import os

os.environ["HARDWARE_MODE"] = "synthetic"
import pytest
from apps.common.schemas import CollectionConfig


@pytest.fixture
def config():
    return CollectionConfig(
        session_id="test_session",
        sender={"logical_name": "tx_main", "port": "synthetic://tx"},
        receivers=[
            {"logical_name": "rx_left", "port": "synthetic://rx0"},
            {"logical_name": "rx_right", "port": "synthetic://rx1"},
        ],
        camera={"device": "synthetic://camera", "width": 320, "height": 240, "fps": 20},
        synthetic_loss=0.05,
        duration_seconds=1,
    )
