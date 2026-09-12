import csv
import io
import pytest
from apps.hardware_service.app.csi import parse_csi, SequenceTracker, LEGACY, COMPACT


def line(fields=LEGACY, **overrides):
    values = {k: 0 for k in fields}
    values.update(
        type="CSI_DATA", mac="1a:00:00:00:00:00", data="[ 2, 7, -4, 9 ]", len=4
    )
    values.update(overrides)
    out = io.StringIO()
    csv.writer(out).writerow([values[k] for k in fields])
    return out.getvalue().strip()


@pytest.mark.parametrize("layout", [LEGACY, COMPACT])
def test_preserve_both_layouts_and_original_array(layout):
    raw = line(layout, **({"seq": 123} if layout == COMPACT else {"id": 123}))
    parsed = parse_csi(raw)
    assert parsed["tx_seq"] == 123
    assert parsed["data"] == "[ 2, 7, -4, 9 ]"
    assert parsed["raw_line"] == raw
    assert parsed["esp_local_timestamp"] == 0
    assert parsed["first_word"] == 0


def test_int16_gain_values_and_signed_printf():
    p = parse_csi(
        line(id=-1, local_timestamp=-10, data="[300,-300,1000,-1000]", first_word=1)
    )
    assert p["tx_seq"] == 2**32 - 1
    assert p["esp_local_timestamp"] == 2**32 - 10
    assert p["first_word"] == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"len": 6},
        {"data": "[1,2,3]", "len": 3},
        {"data": "[true,1]", "len": 2},
        {"mac": "bad"},
        {"first_word": 2},
        {"data": "[32768,0]", "len": 2},
        {"data": "{}"},
        {"id": 2**33},
    ],
)
def test_reject_malformed(changes):
    with pytest.raises(ValueError):
        parse_csi(line(**changes))


def test_logs_and_unknown_layout():
    assert parse_csi("I (123) csi_recv: start") is None
    with pytest.raises(ValueError):
        parse_csi("CSI_DATA,1,2")


def test_sequence_loss_duplicate_reset_and_wrap():
    t = SequenceTracker()
    for n in [2**32 - 2, 2**32 - 1, 0, 3, 3, 1, 2]:
        t.update(n)
    assert t.snapshot() == dict(
        latest_tx_seq=2, sequence_gaps=2, duplicates=1, backwards_or_resets=1, wraps=1
    )


def test_config_traversal_and_duplicate_ports(config):
    from pydantic import ValidationError
    from apps.common.schemas import CollectionConfig, FlashRequest

    for changes in [
        {"session_id": "../escape"},
        {"receivers": [config.receivers[0].model_dump()] * 2},
        {"expected_rate_hz": float("nan")},
        {"camera": {"device": "http://evil"}},
    ]:
        with pytest.raises(ValidationError):
            CollectionConfig(**{**config.model_dump(), **changes})
    with pytest.raises(ValidationError):
        FlashRequest(port="/dev/ttyUSB0;reboot", firmware="blink", target="esp32")
