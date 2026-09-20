"""The alert bot: token handling, message shape, and refusing to send blind."""

import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dc_bot"))

import alert as alert_module  # noqa: E402
import bot as bot_module  # noqa: E402

TOKEN = "test-token-not-a-real-one"


@pytest.fixture
def offline(monkeypatch):
    """No gateway connection in tests: presence is network, not logic."""

    async def stay_put(token):
        return None

    monkeypatch.setattr(bot_module, "presence", stay_put)
    monkeypatch.setenv("DISCORD_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("DISCORD_CHANNEL_ID", "42")


def discord(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_token_and_channel_come_from_the_environment(monkeypatch):
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    monkeypatch.delenv("DISCORD_CHANNEL_ID", raising=False)
    assert alert_module.bot_token() == ""
    # No token means no send, rather than an unauthenticated request.
    assert alert_module.send_discord_alert("falling") is False
    assert alert_module.default_channel() == alert_module.DEFAULT_CHANNEL_ID
    monkeypatch.setenv("DISCORD_BOT_TOKEN", f"  {TOKEN}  ")
    monkeypatch.setenv("DISCORD_CHANNEL_ID", "99")
    assert alert_module.bot_token() == TOKEN
    assert alert_module.default_channel() == "99"
    # The source file carries no token of its own.
    source = (Path(__file__).resolve().parents[2] / "dc_bot/alert.py").read_text()
    assert "TOKEN =" not in source


def test_messages_keep_the_bot_wording_and_add_detail():
    assert alert_module.message_for("falling") == alert_module.EVENT_MESSAGES["falling"]
    detailed = alert_module.message_for("falling", "replay of session x")
    assert detailed.startswith(alert_module.EVENT_MESSAGES["falling"])
    assert detailed.endswith("replay of session x")
    assert "未知事件通知：dancing" in alert_module.message_for("dancing")


def test_the_walking_total_is_worded_by_the_bot_around_the_count():
    assert alert_module.message_for("walking_total", seconds=137.4).startswith(
        "🚶 使用者已經走了 137 秒"
    )
    summary = alert_module.message_for("walking_total", "replay of session x", 12.4)
    assert summary.startswith("🚶 使用者已經走了 12 秒")
    assert summary.endswith("replay of session x")
    # A count that never arrives still reads as a sentence, not a template.
    assert "{" not in alert_module.message_for("walking_total")


def test_alert_endpoint_posts_to_discord(offline):
    sent = []

    def handler(request):
        sent.append(
            dict(
                url=str(request.url),
                auth=request.headers["authorization"],
                body=request.read().decode(),
            )
        )
        return httpx.Response(200, json={"id": "1"})

    with TestClient(bot_module.app) as client:
        client.app.state.client = discord(handler)
        result = client.post(
            "/alert", json={"event": "falling", "detail": "still for 3.0s"}
        )
        assert result.status_code == 200, result.text
        assert result.json()["channel_id"] == "42"
        assert "still for 3.0s" in result.json()["content"]
        assert client.get("/health").json()["sent"] == 1
    assert sent[0]["url"] == "https://discord.com/api/v10/channels/42/messages"
    assert sent[0]["auth"] == f"Bot {TOKEN}"
    assert alert_module.EVENT_MESSAGES["falling"] in sent[0]["body"]


def test_alert_endpoint_sends_the_walking_total_with_its_count(offline):
    sent = []

    def handler(request):
        sent.append(request.read().decode())
        return httpx.Response(200, json={"id": "1"})

    with TestClient(bot_module.app) as client:
        client.app.state.client = discord(handler)
        result = client.post(
            "/alert",
            json={"event": "walking_total", "detail": "live capture", "seconds": 42.4},
        )
        assert result.status_code == 200, result.text
        assert result.json()["content"].startswith("🚶 使用者已經走了 42 秒")
        # The count is a duration, not free text.
        assert (
            client.post(
                "/alert", json={"event": "walking_total", "seconds": -1}
            ).status_code
            == 422
        )
    assert "使用者已經走了 42 秒" in json.loads(sent[0])["content"]


def test_alert_endpoint_reports_a_missing_token_and_a_refusing_discord(
    offline, monkeypatch
):
    def refuse(request):
        return httpx.Response(403, text="Missing Access")

    with TestClient(bot_module.app) as client:
        client.app.state.client = discord(refuse)
        assert client.post("/alert", json={"event": "falling"}).status_code == 502
        assert client.post("/alert", json={}).status_code == 422
        assert (
            client.post("/alert", json={"event": "falling", "who": "me"}).status_code
            == 422
        )
        monkeypatch.delenv("DISCORD_BOT_TOKEN")
        result = client.post("/alert", json={"event": "falling"})
        assert result.status_code == 503 and "DISCORD_BOT_TOKEN" in result.text


def test_health_reports_whether_the_bot_is_configured(offline):
    with TestClient(bot_module.app) as client:
        health = client.get("/health").json()
    assert health["status"] == "ok" and health["token_configured"] is True
    assert "connected" in health and "last_error" in health
