"""Alerting: a fall followed by stillness, and how long the person walked.

Both are judged on the source clock: every time is a source time — seconds
along the live capture or the *recording* being replayed — so a 4x replay
raises exactly the same alerts, and totals the same walking seconds, at the
same points of the recording as playing it back at 1x would.
"""

import os

import httpx

FALL = "falling"
STILL = "static"
WALK = "walking"
# A fall matters when the person then stays down: two seconds of stillness
# inside the four seconds after the fall.
WATCH_SECONDS = 4.0
STILLNESS_SECONDS = 2.0
# One alert per event, not one per prediction that still satisfies the rule.
COOLDOWN_SECONDS = 30.0
# No prediction stands for more time than the window it was made from, so a
# gap in predictions cannot be credited to the walk that preceded it.
WALK_WINDOW_SECONDS = 2.0


class FallWatcher:
    """Feed it fused predictions in order; it returns an alert when one is due."""

    def __init__(
        self,
        watch_seconds=WATCH_SECONDS,
        stillness_seconds=STILLNESS_SECONDS,
        cooldown_seconds=COOLDOWN_SECONDS,
    ):
        self.watch = watch_seconds
        self.stillness = stillness_seconds
        self.cooldown = cooldown_seconds
        self.fell_at = None
        self.still_since = None
        self.alerted_at = None

    def observe(self, label, at):
        """Return (fell_at, still_seconds) when this prediction completes a fall.

        Stillness is the span from the first Static prediction after the fall to
        this one, so predictions of other actions in between do not reset it.
        """
        name = (label or "").strip().lower()
        if name == FALL:
            # A new fall restarts the watch; stillness must follow *this* one.
            self.fell_at, self.still_since = at, None
            return None
        if self.fell_at is None:
            return None
        if at - self.fell_at > self.watch:
            self.fell_at, self.still_since = None, None
            return None
        if name != STILL:
            # A stray other action does not cancel the stillness: the span runs
            # from the first Static prediction after the fall to the latest one.
            return None
        if self.still_since is None:
            self.still_since = at
            return None
        held = at - self.still_since
        if held + 1e-9 < self.stillness:
            return None
        if self.alerted_at is not None and at - self.alerted_at < self.cooldown:
            return None
        self.alerted_at = at
        fell_at, self.fell_at, self.still_since = self.fell_at, None, None
        return fell_at, held


class WalkWatcher:
    """Total the time spent walking, from the run's start until it stops.

    A prediction describes the span until the next one, so Walking is credited
    forward: a walk seen at 10s and still walking at 11s adds one second. A gap
    longer than the model's window means predictions stopped rather than that
    the walk continued, so no more than one window is credited at a time.
    """

    def __init__(self, window_seconds=WALK_WINDOW_SECONDS):
        self.window = max(float(window_seconds), 0.0)
        self.total = 0.0
        self.last_at = None
        self.walking = False

    def observe(self, label, at):
        """Feed it every fused prediction in order; returns the total so far."""
        if self.walking and self.last_at is not None and at > self.last_at:
            self.total += min(at - self.last_at, self.window)
        self.last_at = at
        self.walking = (label or "").strip().lower() == WALK
        return self.total


def describe(fell_at, still_seconds, options, receivers):
    """Say plainly what was seen, and never pass a replay off as live."""
    source = (
        f"replay of recorded session {options.replay_session_id}"
        if options.source == "replay"
        else "live capture"
    )
    return (
        f"偵測：跌倒後靜止 {still_seconds:.1f} 秒 "
        f"(Falling at {fell_at:.1f}s, {source}; "
        f"receivers {', '.join(receivers) or 'none'})"
    )


def describe_walk(seconds, options, receivers):
    """The context line under the walking total; a replay says it is one."""
    source = (
        f"replay of recorded session {options.replay_session_id}"
        if options.source == "replay"
        else "live capture"
    )
    return (
        f"統計：本次部署走動時間 {seconds:.1f} 秒 "
        f"({source}; receivers {', '.join(receivers) or 'none'})"
    )


def bot_health(timeout=3):
    """Ask the bot whether it is actually online, so the UI can say so early."""
    url = os.getenv("NOTIFIER_URL", "http://dc-bot:8003")
    response = httpx.get(f"{url}/health", timeout=timeout)
    response.raise_for_status()
    health = response.json()
    if not health.get("token_configured"):
        raise RuntimeError(
            "Discord alerts are unconfigured: set DISCORD_BOT_TOKEN in .env and restart the bot"
        )
    if not health.get("connected"):
        raise RuntimeError(
            f"The alert bot is not online: {health.get('last_error') or 'connecting'}"
        )
    return health


def notify(event, detail, seconds=None, timeout=5):
    """Post to the alert bot; never let a notification failure stop inference."""
    url = os.getenv("NOTIFIER_URL", "http://dc-bot:8003")
    body = dict(event=event, detail=detail)
    if seconds is not None:
        # The bot words the sentence; the count is ours to supply.
        body["seconds"] = seconds
    response = httpx.post(f"{url}/alert", json=body, timeout=timeout)
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"Alert bot returned HTTP {response.status_code}: {detail}")
    return response.json()
