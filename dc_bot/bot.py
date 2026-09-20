"""The alert bot: a gateway connection so it shows up online, and an alert API.

`make up` starts this alongside the collector. Holding a Discord gateway session
is what makes a bot appear online — REST calls alone leave it grey — so this
service keeps one open, identifying and heartbeating, and reconnects on its own.
Deployment posts to /alert when it detects a fall, and once more with the
walking total when a run stops.
"""

import asyncio
import contextlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
import websockets
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from alert import API, bot_token, default_channel, message_for

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dc_bot")

GATEWAY = "wss://gateway.discord.gg/?v=10&encoding=json"
# No privileged intents: this bot only posts messages and holds a presence.
INTENTS = 0
AUTHENTICATION_FAILED = {4004, 4010, 4011, 4012, 4013, 4014}

state = dict(
    connected=False,
    ready_at=None,
    user=None,
    reconnects=0,
    last_error=None,
    token_configured=False,
    sent=0,
    last_sent_at=None,
)


class AlertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: str = Field(min_length=1, max_length=40)
    detail: str = Field(default="", max_length=1500)
    channel_id: str = Field(default="", max_length=40)
    # Counted events (the walking total) word their sentence around this.
    seconds: float | None = Field(default=None, ge=0, le=86400)


async def heartbeat(socket, interval, sequence):
    # Discord closes a session that stops heartbeating, which drops the bot
    # back to offline; jitter only the first beat, as the protocol asks.
    await asyncio.sleep(interval * 0.5)
    while True:
        await socket.send(json.dumps({"op": 1, "d": sequence[0]}))
        await asyncio.sleep(interval)


async def session(token):
    """One gateway session: identify, heartbeat, stay online until it drops."""
    async with websockets.connect(GATEWAY, max_size=2**20) as socket:
        hello = json.loads(await socket.recv())
        interval = hello["d"]["heartbeat_interval"] / 1000
        sequence = [None]
        await socket.send(
            json.dumps(
                {
                    "op": 2,
                    "d": {
                        "token": token,
                        "intents": INTENTS,
                        "properties": {
                            "os": "linux",
                            "browser": "csi-collection-lab",
                            "device": "csi-collection-lab",
                        },
                        "presence": {
                            "status": "online",
                            "afk": False,
                            "since": None,
                            "activities": [{"name": "CSI fall alerts", "type": 3}],
                        },
                    },
                }
            )
        )
        beat = asyncio.create_task(heartbeat(socket, interval, sequence))
        try:
            async for raw in socket:
                payload = json.loads(raw)
                if payload.get("s") is not None:
                    sequence[0] = payload["s"]
                if payload["op"] == 0 and payload.get("t") == "READY":
                    user = payload["d"]["user"]
                    state.update(
                        connected=True,
                        ready_at=time.time(),
                        user=f"{user['username']}#{user.get('discriminator', '0')}",
                        last_error=None,
                    )
                    logger.info("Discord bot online as %s", state["user"])
                elif payload["op"] in {7, 9}:
                    return  # Server asked for a fresh session.
        finally:
            beat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await beat
            state.update(connected=False)


async def presence(token):
    delay = 2
    while True:
        try:
            await session(token)
            delay = 2
        except asyncio.CancelledError:
            raise
        except websockets.exceptions.ConnectionClosed as error:
            if error.code in AUTHENTICATION_FAILED:
                # Retrying a rejected token just gets the bot rate limited.
                state.update(
                    connected=False,
                    last_error=f"Discord rejected the bot token (close {error.code}); check DISCORD_BOT_TOKEN",
                )
                logger.error(state["last_error"])
                return
            state.update(connected=False, last_error=str(error))
        except Exception as error:
            state.update(connected=False, last_error=str(error))
            logger.warning("Gateway connection lost: %s", error)
        state["reconnects"] += 1
        await asyncio.sleep(delay)
        delay = min(delay * 2, 60)


@asynccontextmanager
async def lifespan(app):
    token = bot_token()
    state["token_configured"] = bool(token)
    task = None
    if token:
        task = asyncio.create_task(presence(token))
    else:
        state["last_error"] = "DISCORD_BOT_TOKEN is not set; alerts are disabled"
        logger.warning(state["last_error"])
    app.state.client = httpx.AsyncClient(timeout=10)
    try:
        yield
    finally:
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await app.state.client.aclose()


app = FastAPI(title="CSI Lab Discord alerts", lifespan=lifespan)


@app.get("/health")
def health():
    return dict(status="ok", **state)


@app.post("/alert")
async def alert(body: AlertRequest):
    token = bot_token()
    if not token:
        raise HTTPException(503, "DISCORD_BOT_TOKEN is not set; alerts are disabled")
    channel = body.channel_id or default_channel()
    content = message_for(body.event, body.detail, body.seconds)
    try:
        response = await app.state.client.post(
            f"{API}/channels/{channel}/messages",
            headers={"Authorization": f"Bot {token}"},
            json={"content": content},
        )
    except httpx.HTTPError as error:
        raise HTTPException(502, f"Discord unreachable: {error}") from error
    if response.is_error:
        raise HTTPException(
            502,
            f"Discord refused the message (HTTP {response.status_code}): {response.text}",
        )
    state.update(sent=state["sent"] + 1, last_sent_at=time.time())
    logger.info("Sent %s alert to channel %s", body.event, channel)
    return dict(sent=True, event=body.event, channel_id=channel, content=content)
