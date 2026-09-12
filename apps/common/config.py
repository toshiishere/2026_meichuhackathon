import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.getenv("DATA_DIR", str(ROOT / "data")))
MODE = os.getenv("HARDWARE_MODE", "real")
if MODE not in {"real", "synthetic"}:
    raise RuntimeError("HARDWARE_MODE must be real or synthetic")
DEFAULTS = yaml.safe_load((ROOT / "configs/collection.yaml").read_text())
SCHEMA_VERSION = "1.0"
