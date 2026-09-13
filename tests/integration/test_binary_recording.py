import base64
import csv
import io
import json
import zstandard
from apps.hardware_service.app import recorder
from apps.hardware_service.app.csi import parse_csi
from tests.test_csi_binary import binary_frame


def test_binary_recording_with_video_preserves_transport_and_counts_firmware_loss(
    config, tmp_path, monkeypatch
):
    original = recorder.SerialSource

    class BinarySource(original):
        def read(self, stop):
            value = super().read(stop)
            if value is None:
                return None
            row = parse_csi(value[0])
            seq = row["tx_seq"]
            return binary_frame(seq=seq, drops=10 if seq < 30 else 13), value[1]

    monkeypatch.setattr(recorder, "SerialSource", BinarySource)
    config.synthetic_loss = 0
    result = recorder.Recorder(config, tmp_path).run()
    assert result["status"] == "complete", result["errors"]
    assert result["quality"] == "degraded"
    assert all(
        r["firmware_queue_drops"] == 3 for r in result["statistics"]["receivers"]
    )
    path = tmp_path / "sessions/test_session/raw/csi_rx_left.csv.zst"
    with path.open("rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as stream:
        rows = list(csv.DictReader(io.TextIOWrapper(stream)))
    assert len(rows) >= 80
    for row in rows:
        packet = parse_csi(base64.b64decode(row["raw_binary_base64"]))
        assert packet["tx_seq"] == int(row["tx_seq"])
        assert json.loads(packet["data"]) == json.loads(row["data"])
        assert row["sample_representation"] == "raw_int8"
        assert int(row["host_timestamp_ns"]) >= result["start_timestamp_ns"]
