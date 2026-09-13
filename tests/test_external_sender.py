import pytest
from pydantic import ValidationError
from apps.common.schemas import CollectionConfig
from apps.hardware_service.app import checks


def test_receiver_only_preflight_never_requires_sender_port(
    config, tmp_path, monkeypatch
):
    data = config.model_dump(exclude={"sender"})
    external = CollectionConfig(**data)
    assert external.sender is None
    original = checks.validate_port
    checked = []

    def receiver_only(port):
        assert port != "synthetic://tx", "Battery sender has no USB port"
        checked.append(port)
        return original(port)

    monkeypatch.setattr(checks, "validate_port", receiver_only)
    result = checks.preflight(external, tmp_path)
    assert result["passed"], result
    assert result["sender"] is None
    assert result["sender_connection"] == "external"
    assert set(checked) == {r.port for r in config.receivers}
    with pytest.raises(ValidationError, match="unique name and port"):
        CollectionConfig(**{**data, "receivers": [data["receivers"][0]] * 2})


def test_explicit_sender_still_checks_identity(config, tmp_path, monkeypatch):
    config.sender.identity = "wrong-board"
    with pytest.raises(ValueError, match="identity changed"):
        checks.preflight(config, tmp_path)
