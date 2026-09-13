"""Compile preserved firmware without a connected board; never flashes hardware."""

import argparse
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apps.common.schemas import FlashRequest
from apps.hardware_service.app.firmware import flash

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("firmware", choices=["csi-send", "csi-recv", "blink"])
parser.add_argument(
    "--target", required=True, choices=["esp32", "esp32c3", "esp32s3", "esp32c6"]
)
parser.add_argument("--gpio", type=int)
parser.add_argument("--led-type", choices=["gpio", "rgb"], default="gpio")
parser.add_argument("--csi-transport", choices=["auto", "usb", "uart"], default="auto")
args = parser.parse_args()
request = FlashRequest(
    port="/dev/ttyUSB0",
    target=args.target,
    firmware=args.firmware,
    operation="build",
    gpio=args.gpio,
    led_type=args.led_type,
    csi_transport=args.csi_transport,
)
print(flash(request, print, threading.Event()))
