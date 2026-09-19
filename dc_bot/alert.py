"""Discord alert messages.

The bot token is read from the environment (DISCORD_BOT_TOKEN, set in .env and
passed through docker-compose). It is never stored in this file: a token in the
repository is a token anyone who reads the repository can use.
"""

import logging
import os

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API = "https://discord.com/api/v10"
DEFAULT_CHANNEL_ID = "1550768488086380554"

EVENT_MESSAGES = {
    "falling": "🚨 警告：年長者跌倒了，是否呼叫救護車？",
    "sitting": "ℹ️ 狀態更新：年長者目前正在坐著。",
    "walking": "ℹ️ 狀態更新：年長者目前正在走動。",
    "abnormal": "⚠️ 警告：偵測到異常活動，請確認狀況！",
}


def bot_token():
    return os.getenv("DISCORD_BOT_TOKEN", "").strip()


def default_channel():
    return os.getenv("DISCORD_CHANNEL_ID", "").strip() or DEFAULT_CHANNEL_ID


def message_for(event_type, detail=""):
    message = EVENT_MESSAGES.get(event_type, f"未知事件通知：{event_type}")
    return f"{message}\n{detail}" if detail else message


def send_discord_alert(event_type: str, channel_id: str = "", detail: str = "") -> bool:
    """Send an alert message to a Discord channel based on the event type."""
    token, channel = bot_token(), channel_id or default_channel()
    if not token:
        logger.error("DISCORD_BOT_TOKEN is not set; cannot send Discord messages.")
        return False
    if not channel:
        logger.error("Discord channel ID is not configured.")
        return False
    content = message_for(event_type, detail)
    try:
        response = httpx.post(
            f"{API}/channels/{channel}/messages",
            headers={"Authorization": f"Bot {token}"},
            json={"content": content},
            timeout=10,
        )
        response.raise_for_status()
        logger.info("Successfully sent message to Discord: %s", content)
        return True
    except httpx.HTTPError as error:
        logger.error("Failed to send Discord message: %s", error)
        response = getattr(error, "response", None)
        if response is not None:
            logger.error("Error details: %s", response.text)
        return False


if __name__ == "__main__":
    import sys

    test_event = sys.argv[1] if len(sys.argv) > 1 else "falling"
    print(f"Testing event dispatch: {test_event}")
    send_discord_alert(test_event)
