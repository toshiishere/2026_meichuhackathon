import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apps.common.config import DATA
from apps.common.storage import rebuild_manifest

print(rebuild_manifest(DATA))
