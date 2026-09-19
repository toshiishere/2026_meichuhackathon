import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Discord Bot Token
TOKEN = "MTU1MDY4ODcyOTg4MDg2MjcyMA.Gr-a4A.6XvmnPvcKekAEQL3Ko7mlN8AN75EvlDvMKYZ4g"
DEFAULT_CHANNEL_ID = "1550768488086380554"

EVENT_MESSAGES = {
    "falling": "🚨 警告：年長者跌倒了，是否呼叫救護車？",
    "sitting": "ℹ️ 狀態更新：年長者目前正在坐著。",
    "walking": "ℹ️ 狀態更新：年長者目前正在走動。",
    "abnormal": "⚠️ 警告：偵測到異常活動，請確認狀況！"
}

def send_discord_alert(event_type: str, channel_id: str = DEFAULT_CHANNEL_ID) -> bool:
    """Send an alert message to a Discord channel based on the event type."""
    if not channel_id:
        logger.error("Discord channel ID is not configured.")
        return False

    message_content = EVENT_MESSAGES.get(event_type, f"未知事件通知：{event_type}")
    
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    headers = {
        "Authorization": f"Bot {TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "content": message_content
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        logger.info(f"Successfully sent message to Discord: {message_content}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send Discord message: {e}")
        if e.response is not None:
            logger.error(f"Error details: {e.response.text}")
        return False

if __name__ == "__main__":
    import sys
    test_event = sys.argv[1] if len(sys.argv) > 1 else "falling"
    print(f"Testing event dispatch: {test_event}")
    send_discord_alert(test_event)

