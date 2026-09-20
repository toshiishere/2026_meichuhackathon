"""Alerting: the fall rule, the walking total, and what a deployment sends."""

import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from apps.common.schemas import DeployRequest, NotifyRequest
from apps.training_service.app import deploy, main
from apps.training_service.app.alerts import (
    FallWatcher,
    WalkWatcher,
    describe,
    describe_walk,
)
from test_deploy import (
    RecordingPredictor,
    model_fixture,
    multi_receiver_session,
    wait_for,
)


def feed(watcher, predictions):
    return [watcher.observe(label, at) for label, at in predictions]


def test_a_fall_alerts_only_once_stillness_holds_for_two_seconds():
    watcher = FallWatcher()
    # Stillness spans from the first Static prediction after the fall, so it
    # only reaches two seconds once three one-second predictions have landed.
    results = feed(
        watcher,
        [
            ("Walking", 9.0),
            ("Falling", 10.0),
            ("Static", 11.0),
            ("Static", 12.0),
            ("Static", 13.0),
            ("Static", 14.0),
        ],
    )
    assert results[:4] == [None] * 4
    assert results[4] == (10.0, 2.0)
    # The same fall does not alert again on the next still prediction.
    assert results[5] is None


def test_a_stray_action_does_not_cancel_the_stillness():
    interrupted = feed(
        FallWatcher(),
        [
            ("Falling", 0.0),
            ("Static", 1.0),
            ("Walking", 2.0),
            ("Static", 3.0),
        ],
    )
    # The Walking prediction between the two Static ones does not reset the span.
    assert interrupted[:3] == [None] * 3
    assert interrupted[3] == (0.0, 2.0)


def test_stillness_must_still_fall_inside_the_four_second_window():
    too_late = feed(
        FallWatcher(),
        [
            ("Falling", 0.0),
            ("Walking", 1.0),
            ("Walking", 2.0),
            ("Static", 3.0),
            ("Static", 4.0),
            ("Static", 5.0),
        ],
    )
    assert too_late == [None] * 6  # Stillness started too late to span 2s by 4s.

    # A second fall restarts the watch rather than reusing the first one.
    watcher = FallWatcher()
    assert (
        feed(
            watcher,
            [
                ("Falling", 0.0),
                ("Static", 1.0),
                ("Falling", 2.0),
                ("Static", 3.0),
            ],
        )
        == [None] * 4
    )
    assert watcher.observe("Static", 5.0) == (2.0, 2.0)


def test_alerting_depends_on_source_time_not_on_replay_speed():
    """The same recording alerts identically at 1x and at 4x."""
    predictions = [
        ("Falling", 20.0),
        ("Static", 21.0),
        ("Static", 22.0),
        ("Static", 23.0),
    ]
    slow = FallWatcher()
    fast = FallWatcher()
    for label, at in predictions:
        slow.observe(label, at)
        time.sleep(0.01)  # Wall-clock pacing differs; source seconds do not.
    results = [fast.observe(label, at) for label, at in predictions]
    assert results[-1] == (20.0, 2.0)
    assert slow.alerted_at == fast.alerted_at == 23.0


def test_walking_time_adds_up_across_the_whole_run():
    walker = WalkWatcher(window_seconds=2.0)
    # Each prediction stands until the next one, so the two Walking seconds
    # before the person stops are counted and the Static span is not.
    for label, at in [
        ("Static", 0.0),
        ("Walking", 1.0),
        ("Walking", 2.0),
        ("Static", 3.0),
        ("Static", 4.0),
        ("Walking", 5.0),
    ]:
        walker.observe(label, at)
    assert walker.total == pytest.approx(2.0)
    # The walk still running at the last prediction is counted from there on.
    assert walker.observe("Walking", 6.0) == pytest.approx(3.0)
    assert walker.observe("Static", 7.0) == pytest.approx(4.0)


def test_a_gap_in_predictions_is_not_credited_to_the_walk():
    walker = WalkWatcher(window_seconds=2.0)
    walker.observe("Walking", 0.0)
    # Predictions stopped for a minute; a walk is only ever worth the window
    # the prediction was made from.
    assert walker.observe("Walking", 60.0) == pytest.approx(2.0)
    assert walker.observe("Static", 61.0) == pytest.approx(3.0)


def test_the_walking_total_counts_source_seconds_not_wall_clock():
    """The same recording totals the same seconds at 1x and at 4x."""
    predictions = [("Walking", 1.0), ("Walking", 2.0), ("Static", 3.0)]
    slow, fast = WalkWatcher(), WalkWatcher()
    for label, at in predictions:
        slow.observe(label, at)
        time.sleep(0.01)  # Wall-clock pacing differs; source seconds do not.
    for label, at in predictions:
        fast.observe(label, at)
    assert slow.total == fast.total == pytest.approx(2.0)


def test_the_walking_summary_says_which_source_it_counted(tmp_path):
    options = DeployRequest(
        model_session_id="trained",
        source="replay",
        replay_session_id="recorded",
        replay_receivers=["left"],
        replay_speed=4,
    )
    detail = describe_walk(12.5, options, ["left", "right"])
    assert "replay of recorded session recorded" in detail and "12.5" in detail
    live = DeployRequest(
        model_session_id="trained",
        source="live",
        receivers=[dict(logical_name="rx", port="synthetic://rx0")],
    )
    assert "live capture" in describe_walk(3.0, live, ["rx"])


def test_a_replay_alert_says_it_is_a_replay(tmp_path):
    options = DeployRequest(
        model_session_id="trained",
        source="replay",
        replay_session_id="recorded",
        replay_receivers=["left"],
        replay_speed=4,
    )
    detail = describe(12.5, 2.0, options, ["left", "right"])
    assert "replay of recorded session recorded" in detail
    assert "12.5" in detail and "2.0" in detail
    live = DeployRequest(
        model_session_id="trained",
        source="live",
        receivers=[dict(logical_name="rx", port="synthetic://rx0")],
    )
    assert "live capture" in describe(1.0, 2.0, live, ["rx"])


class FallingPredictor(RecordingPredictor):
    """Falling once, then Static: the fall-then-stillness pattern."""

    def __init__(self, *args):
        super().__init__(*args)
        self.calls = 0

    def scores(self, windows):
        self.calls += 1
        # classes are ["Falling", "Static"]; first real window falls, rest still.
        row = [1.0, 0.0] if self.calls == 2 else [0.0, 1.0]
        import numpy as np

        return np.array([row] * len(windows)), 1.0


def test_deployment_sends_one_alert_for_a_replayed_fall(tmp_path, monkeypatch):
    session = multi_receiver_session(tmp_path, names=("left", "right"))
    # Relabel the model so the classes carry the actions the rule looks for.
    import json
    from apps.common.storage import atomic_json

    run = session / "train/runs" / ("a" * 32)
    atomic_json(
        run / "classes.json",
        dict(
            class_names=["Falling", "Static"], class_to_idx={"Falling": 0, "Static": 1}
        ),
    )
    monkeypatch.setattr(deploy, "Predictor", FallingPredictor)
    posted = []

    def fake_notify(event, detail, timeout=5):
        posted.append((event, detail))
        return dict(sent=True)

    monkeypatch.setattr(deploy, "notify", fake_notify)
    manager = deploy.Deployment(tmp_path, threading.Lock(), main.session_path)
    manager.start(
        DeployRequest(
            model_session_id=session.name,
            source="replay",
            replay_session_id=session.name,
            replay_receivers=["left", "right"],
            replay_speed=4,
            notify=True,
        )
    )
    try:
        wait_for(lambda: manager.snapshot()["status"] == "completed", timeout=20)
    finally:
        manager.close()
    state = manager.snapshot()
    assert len(state["falls"]) == 1, state["falls"]
    fall = state["falls"][0]
    assert fall["still_seconds"] >= 2
    wait_for(lambda: manager.snapshot()["falls"][0]["notified"])
    assert [event for event, _ in posted] == ["falling"]
    assert "replay of recorded session" in posted[0][1]


def test_detection_without_sending_when_alerts_are_switched_off(tmp_path, monkeypatch):
    session = multi_receiver_session(tmp_path, names=("left",))
    from apps.common.storage import atomic_json

    run = session / "train/runs" / ("a" * 32)
    atomic_json(
        run / "classes.json",
        dict(
            class_names=["Falling", "Static"], class_to_idx={"Falling": 0, "Static": 1}
        ),
    )
    monkeypatch.setattr(deploy, "Predictor", FallingPredictor)

    def refuse(*args, **kwargs):
        raise AssertionError("no alert may leave the machine while notify is off")

    monkeypatch.setattr(deploy, "notify", refuse)
    manager = deploy.Deployment(tmp_path, threading.Lock(), main.session_path)
    manager.start(
        DeployRequest(
            model_session_id=session.name,
            source="replay",
            replay_session_id=session.name,
            replay_receivers=["left"],
            replay_speed=4,
            notify=False,
        )
    )
    try:
        wait_for(lambda: manager.snapshot()["status"] == "completed", timeout=20)
    finally:
        manager.close()
    state = manager.snapshot()
    # The fall is still detected and reported in the UI, just never sent.
    assert state["falls"] and not state["falls"][0]["notified"]
    assert state["notify"] is False


class WalkingPredictor(RecordingPredictor):
    """Always Walking, so the whole replay counts toward the walking total."""

    def scores(self, windows):
        import numpy as np

        # classes are ["Static", "Walking"].
        return np.array([[0.1, 0.9]] * len(windows)), 1.0


def run_walking_replay(tmp_path, monkeypatch, notify, sender):
    session = multi_receiver_session(tmp_path, names=("left",))
    monkeypatch.setattr(deploy, "Predictor", WalkingPredictor)
    monkeypatch.setattr(deploy, "notify", sender)
    manager = deploy.Deployment(tmp_path, threading.Lock(), main.session_path)
    manager.start(
        DeployRequest(
            model_session_id=session.name,
            source="replay",
            replay_session_id=session.name,
            replay_receivers=["left"],
            replay_speed=4,
            notify=notify,
        )
    )
    try:
        wait_for(lambda: manager.snapshot()["status"] == "completed", timeout=20)
    finally:
        manager.close()
    return manager


def test_a_finished_deployment_reports_how_long_the_walking_lasted(
    tmp_path, monkeypatch
):
    posted = []

    def fake_notify(event, detail, seconds=None, timeout=5):
        posted.append((event, detail, seconds))
        return dict(sent=True)

    manager = run_walking_replay(tmp_path, monkeypatch, True, fake_notify)
    state = manager.snapshot()
    # The recording holds eight seconds of CSI, walked from the first window on.
    assert state["walking_seconds"] >= 5
    summary = state["walking"]
    assert summary["seconds"] == state["walking_seconds"]
    wait_for(lambda: manager.snapshot()["walking"]["notified"])
    assert [event for event, _, _ in posted] == ["walking_total"]
    event, detail, seconds = posted[0]
    # The bot words the sentence; the deployment supplies the count and says
    # plainly that these seconds came from a replay.
    assert seconds == summary["seconds"]
    assert "replay of recorded session" in detail
    assert not state["falls"]


def test_the_walking_total_stays_on_the_machine_when_alerts_are_off(
    tmp_path, monkeypatch
):
    def refuse(*args, **kwargs):
        raise AssertionError("no total may leave the machine while notify is off")

    manager = run_walking_replay(tmp_path, monkeypatch, False, refuse)
    state = manager.snapshot()
    # Counted and shown in the UI all the same, just never sent.
    assert state["walking_seconds"] >= 5
    assert state["walking"]["notified"] is False


def test_a_deployment_without_walking_sends_no_total(tmp_path, monkeypatch):
    session = multi_receiver_session(tmp_path, names=("left",))
    from apps.common.storage import atomic_json

    run = session / "train/runs" / ("a" * 32)
    atomic_json(
        run / "classes.json",
        dict(
            class_names=["Falling", "Static"], class_to_idx={"Falling": 0, "Static": 1}
        ),
    )
    monkeypatch.setattr(deploy, "Predictor", FallingPredictor)
    posted = []
    monkeypatch.setattr(
        deploy,
        "notify",
        lambda event, detail, **kwargs: posted.append(event) or dict(sent=True),
    )
    manager = deploy.Deployment(tmp_path, threading.Lock(), main.session_path)
    manager.start(
        DeployRequest(
            model_session_id=session.name,
            source="replay",
            replay_session_id=session.name,
            replay_receivers=["left"],
            replay_speed=4,
            notify=True,
        )
    )
    try:
        wait_for(lambda: manager.snapshot()["status"] == "completed", timeout=20)
    finally:
        manager.close()
    state = manager.snapshot()
    assert state["walking_seconds"] == 0 and state["walking"]["seconds"] == 0
    # A fall is worth a message; nought seconds of walking is not.
    wait_for(lambda: posted == ["falling"])
    assert posted == ["falling"]


def test_notify_toggle_is_validated_and_reaches_the_worker(tmp_path, monkeypatch):
    session = model_fixture(tmp_path)
    monkeypatch.setattr(main, "DATA", tmp_path)
    monkeypatch.setattr(deploy, "bot_health", lambda: dict(connected=True))
    with TestClient(main.app) as client:
        assert client.post("/deploy/notify", json={}).status_code == 422
        assert (
            client.post("/deploy/notify", json={"enabled": "yes please"}).status_code
            == 422
        )
        assert (
            client.post("/deploy/notify", json={"enabled": True}).json()["notify"]
            is True
        )
        assert (
            client.post("/deploy/notify", json={"enabled": False}).json()["notify"]
            is False
        )
    assert NotifyRequest(enabled=True).enabled is True
    assert session.exists()


def test_alert_failure_is_reported_without_stopping_inference(tmp_path, monkeypatch):
    session = multi_receiver_session(tmp_path, names=("left",))
    from apps.common.storage import atomic_json

    run = session / "train/runs" / ("a" * 32)
    atomic_json(
        run / "classes.json",
        dict(
            class_names=["Falling", "Static"], class_to_idx={"Falling": 0, "Static": 1}
        ),
    )
    monkeypatch.setattr(deploy, "Predictor", FallingPredictor)

    def broken(event, detail, timeout=5):
        raise RuntimeError("Alert bot returned HTTP 503: token missing")

    monkeypatch.setattr(deploy, "notify", broken)
    manager = deploy.Deployment(tmp_path, threading.Lock(), main.session_path)
    manager.start(
        DeployRequest(
            model_session_id=session.name,
            source="replay",
            replay_session_id=session.name,
            replay_receivers=["left"],
            replay_speed=4,
            notify=True,
        )
    )
    try:
        wait_for(lambda: manager.snapshot()["status"] == "completed", timeout=20)
        wait_for(lambda: manager.snapshot().get("notify_error"))
    finally:
        manager.close()
    state = manager.snapshot()
    assert "token missing" in state["notify_error"]
    assert state["status"] == "completed"  # Inference ran to the end regardless.


def test_notifier_client_reports_a_refusing_bot(monkeypatch):
    from apps.training_service.app import alerts

    def transport(request):
        assert request.url.path == "/alert"
        return httpx.Response(503, json={"detail": "DISCORD_BOT_TOKEN is not set"})

    def post(url, **kwargs):
        client = httpx.Client(transport=httpx.MockTransport(transport))
        return client.post(url, **{k: v for k, v in kwargs.items() if k != "timeout"})

    monkeypatch.setattr(alerts.httpx, "post", post)
    with pytest.raises(RuntimeError, match="DISCORD_BOT_TOKEN is not set"):
        alerts.notify("falling", "detail")


def test_switching_alerts_on_says_when_the_bot_is_not_online(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATA", tmp_path)

    def offline():
        raise RuntimeError(
            "The alert bot is not online: Discord rejected the bot token"
        )

    monkeypatch.setattr(deploy, "bot_health", offline)
    with TestClient(main.app) as client:
        state = client.post("/deploy/notify", json={"enabled": True}).json()
        assert state["notify"] is True
        assert "not online" in state["notify_error"]
        # Switching alerts back off clears the complaint.
        assert (
            client.post("/deploy/notify", json={"enabled": False}).json()[
                "notify_error"
            ]
            is None
        )
