import React, { useEffect, useRef, useState } from "react";
import { readApiResponse } from "./api";

type Json = Record<string, any>;
const active = (s?: string) =>
  ["starting", "running", "stopping"].includes(s || "");
async function api(path: string, body?: Json) {
  return readApiResponse(
    await fetch(
      `/api/deploy${path}`,
      body === undefined
        ? undefined
        : {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          },
    ),
  );
}

const toggle = (values: string[], value: string, on: boolean) =>
  on
    ? values.includes(value)
      ? values
      : [...values, value]
    : values.filter((v) => v !== value);

export function Deploy({
  boards,
  ports,
  cameras,
  camera,
}: {
  boards: Json[];
  ports: Json[];
  cameras: Json[];
  camera: Json;
}) {
  const [catalog, setCatalog] = useState<Json>({
    models: [],
    sources: [],
    errors: [],
  });
  const [state, setState] = useState<Json>({ status: "idle" });
  const [model, setModel] = useState("");
  const [source, setSource] = useState("replay");
  const [replay, setReplay] = useState("");
  const [replayReceivers, setReplayReceivers] = useState<string[]>([]);
  const [liveReceivers, setLiveReceivers] = useState<string[]>([]);
  const [speed, setSpeed] = useState(1);
  const [baud, setBaud] = useState(921600);
  const [cameraDevice, setCameraDevice] = useState("");
  const [error, setError] = useState("");
  const [serviceError, setServiceError] = useState("");
  const [busy, setBusy] = useState(false);
  const initialized = useRef(false);
  const video = useRef<HTMLVideoElement>(null);
  const [videoError, setVideoError] = useState("");
  // Half of the status round trip: the reported playhead is already that old.
  const latency = useRef(0);
  const playhead = useRef({ position: NaN, at: 0, rate: 1, playing: false });
  const drift = useRef(0);
  const running = active(state.status);
  const receivers = boards.filter(
    (b) =>
      b.role === "csi_receiver" && ports.some((p) => p.identity === b.identity),
  );
  const selectedModel = catalog.models.find(
    (m: Json) => m.session_id === model,
  );
  const sessionReceivers: string[] =
    catalog.sources.find((s: Json) => s.session_id === replay)?.receivers || [];

  async function refreshCatalog() {
    try {
      const data = await api("/catalog");
      setError("");
      setCatalog(data);
      setModel((old) => old || data.models[0]?.session_id || "");
      setReplay((old) => old || data.sources[0]?.session_id || "");
    } catch (e) {
      setError((e as Error).message);
    }
  }
  useEffect(() => {
    void refreshCatalog();
  }, []);
  const chosenFor = useRef("");
  useEffect(() => {
    if (running) return;
    // A different recording starts with every one of its receivers fused, the
    // way training used them; within one recording the boxes are the operator's.
    const fresh = chosenFor.current !== replay;
    chosenFor.current = replay;
    setReplayReceivers((old) => {
      const kept = fresh
        ? []
        : old.filter((name) => sessionReceivers.includes(name));
      return kept.length || !sessionReceivers.length ? kept : sessionReceivers;
    });
  }, [replay, catalog, running]);
  useEffect(() => {
    if (running) return;
    setLiveReceivers((old) => {
      const kept = old.filter((id) => receivers.some((b) => b.identity === id));
      return kept.length ? kept : receivers.map((b) => b.identity);
    });
  }, [boards, ports, running]);
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const began = performance.now();
        const value = await api("/status");
        if (!alive) return;
        latency.current = performance.now() - began;
        setState(value);
        setServiceError("");
        if (!initialized.current && active(value.status) && value.options) {
          const o = value.options;
          setModel(o.model_session_id);
          setSource(o.source);
          setReplay(o.replay_session_id || "");
          setReplayReceivers(o.replay_receivers || []);
          setLiveReceivers(
            (o.receivers || []).map((r: Json) => r.identity).filter(Boolean),
          );
          setSpeed(o.replay_speed);
          setBaud(o.baud_rate);
          setCameraDevice(o.camera?.device || "");
        }
        initialized.current = true;
      } catch (e) {
        if (alive) setServiceError((e as Error).message);
      } finally {
        if (alive) timer = setTimeout(poll, 200);
      }
    }
    void poll();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, []);

  // Recorded video: track the worker's playhead continuously instead of
  // replaying a status sample that is already a round trip old.
  useEffect(() => {
    playhead.current = {
      position: Number(state.video_time_s),
      at: performance.now() - latency.current / 2,
      rate: state.options?.replay_speed || 1,
      playing:
        state.status === "running" &&
        state.video_playing !== false &&
        !serviceError,
    };
  }, [state, serviceError]);
  useEffect(() => {
    const player = video.current;
    if (!player) return;
    const sync = () => {
      const { position, at, rate, playing } = playhead.current;
      if (Number.isFinite(position)) {
        const expected = playing
          ? position + ((performance.now() - at) / 1000) * rate
          : position;
        drift.current = expected - player.currentTime;
        if (Math.abs(drift.current) > (playing ? 0.35 : 0.02)) {
          player.currentTime = expected;
          player.playbackRate = rate;
        } else if (playing) {
          // Absorb small drift through the rate; seeking every poll stutters.
          const trim = Math.max(-0.1, Math.min(0.1, drift.current * 0.5));
          player.playbackRate = Math.max(0.1, rate * (1 + trim));
        } else {
          player.playbackRate = rate;
        }
      }
      if (playing) {
        void player.play().catch((e: DOMException) => {
          if (e.name !== "AbortError")
            setVideoError(
              "Video playback could not start. Check browser autoplay settings.",
            );
        });
      } else {
        player.pause();
      }
    };
    sync();
    const timer = window.setInterval(sync, 100);
    player.addEventListener("loadedmetadata", sync);
    return () => {
      window.clearInterval(timer);
      player.removeEventListener("loadedmetadata", sync);
    };
  }, [state.video_available, state.id]);

  // Live camera: ask for the frame captured at the fused CSI clock — the end
  // of the window every live receiver covers — and report the real offset.
  const [preview, setPreview] = useState<Json | null>(null);
  const align = useRef<number | null>(null);
  const objectUrl = useRef("");
  useEffect(() => {
    align.current =
      state.source_timestamp_ns || state.prediction?.window_end_ns || null;
  }, [state]);
  const liveCamera = !!(state.capture_id && state.options?.camera);
  useEffect(() => {
    if (!liveCamera) {
      setPreview(null);
      return;
    }
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const show = (value: Json | null) => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
      objectUrl.current = value?.url || "";
      setPreview(value);
    };
    async function load() {
      try {
        const timestamp = align.current;
        const response = await fetch(
          `/api/deploy/camera/${state.capture_id}` +
            (timestamp ? `?timestamp_ns=${timestamp}` : ""),
          { cache: "no-store" },
        );
        if (!response.ok) throw new Error(String(response.status));
        const stamp = Number(response.headers.get("X-Host-Timestamp-Ns"));
        const blob = await response.blob();
        if (!alive) return;
        show({
          url: URL.createObjectURL(blob),
          skew:
            timestamp && Number.isFinite(stamp)
              ? (stamp - timestamp) / 1e6
              : null,
        });
      } catch {
        if (alive) show(null);
      } finally {
        if (alive) timer = setTimeout(load, 100);
      }
    }
    void load();
    return () => {
      alive = false;
      clearTimeout(timer);
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
      objectUrl.current = "";
    };
  }, [liveCamera, state.capture_id]);

  async function start() {
    setBusy(true);
    setError("");
    setVideoError("");
    try {
      const selected = liveReceivers
        .map((identity) => ({
          board: receivers.find((x) => x.identity === identity),
          port: ports.find((x) => x.identity === identity),
        }))
        .filter((x) => x.board && x.port);
      if (source === "live" && selected.length !== liveReceivers.length)
        throw new Error("A selected receiver is disconnected; refresh hardware.");
      const selectedCamera = cameras.find((c) => c.device === cameraDevice);
      const cameraConfig =
        source === "live" && cameraDevice
          ? {
              device: cameraDevice,
              width: selectedCamera?.width || camera.width,
              height: selectedCamera?.height || camera.height,
              fps: selectedCamera?.fps || camera.fps,
              fixed_frame_rate: camera.fixed_frame_rate,
            }
          : null;
      const result = await api("/start", {
        model_session_id: model,
        source,
        receivers:
          source === "live"
            ? selected.map(({ board, port }) => ({
                identity: board!.identity,
                logical_name: board!.logical_name,
                port: port!.port,
              }))
            : [],
        replay_session_id: source === "replay" ? replay : null,
        replay_receivers: source === "replay" ? replayReceivers : [],
        replay_speed: speed,
        baud_rate: baud,
        camera: cameraConfig,
      });
      setState(result);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function stop() {
    setBusy(true);
    setError("");
    try {
      setState(await api("/stop", {}));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const prediction = serviceError ? null : state.prediction;
  const signalText: Json = {
    warming_up: "Collecting a full CSI window…",
    waiting_for_supported_csi: "Waiting for supported CSI packets…",
    insufficient_data: "Not enough CSI coverage. Check packet rate and gaps.",
    no_data: "CSI stream paused. Waiting for fresh data…",
  };
  const chosen = source === "replay" ? replayReceivers : liveReceivers;
  return (
    <>
      {error && (
        <div className="alert" role="alert">
          {error}
        </div>
      )}
      {serviceError && (
        <div className="notice warning" role="status">
          {serviceError} Deployment may still be running; reconnect before
          starting again.
        </div>
      )}
      <section className="panel">
        <div className="panel-heading">
          <h2>Deploy a session model</h2>
          <button
            className="secondary"
            disabled={busy}
            onClick={() => void refreshCatalog()}
          >
            Refresh models
          </button>
        </div>
        <p>
          Every selected receiver is read over one shared CSI window, scored by
          the trained model and fused into a single pose — the same way training
          used all receivers. Live capture and recorded replay use the model’s
          own preprocessing. Camera images are for visualization only.
        </p>
        {!catalog.models.length && (
          <p className="notice">
            No trained models available. Fine-tune a completed session in Train
            first.
          </p>
        )}
        <div className="form-grid">
          <label className="field">
            <span>Model session</span>
            <select
              aria-label="Model session"
              disabled={running || busy}
              value={model}
              onChange={(e) => setModel(e.target.value)}
            >
              <option value="">Choose a trained session</option>
              {catalog.models.map((m: Json) => (
                <option key={m.session_id} value={m.session_id}>
                  {m.session_id}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>CSI source</span>
            <select
              aria-label="CSI source"
              value={source}
              disabled={running || busy}
              onChange={(e) => setSource(e.target.value)}
            >
              <option value="replay">Recorded session replay</option>
              <option value="live">Live serial receiver</option>
            </select>
          </label>
          {source === "replay" ? (
            <>
              <label className="field">
                <span>Replay session</span>
                <select
                  aria-label="Replay session"
                  value={replay}
                  disabled={running || busy}
                  onChange={(e) => setReplay(e.target.value)}
                >
                  <option value="">Choose a recording</option>
                  {catalog.sources.map((s: Json) => (
                    <option key={s.session_id}>{s.session_id}</option>
                  ))}
                </select>
              </label>
              <fieldset className="field">
                <span>Replay receivers to fuse</span>
                {sessionReceivers.map((name: string) => (
                  <label className="checkbox" key={name}>
                    <input
                      type="checkbox"
                      aria-label={`Replay receiver ${name}`}
                      checked={replayReceivers.includes(name)}
                      disabled={running || busy}
                      onChange={(e) =>
                        setReplayReceivers((old) =>
                          toggle(old, name, e.target.checked),
                        )
                      }
                    />
                    <span>{name}</span>
                  </label>
                ))}
                {!sessionReceivers.length && (
                  <span className="subtle">No receivers in this recording</span>
                )}
              </fieldset>
              <label className="field">
                <span>Replay speed</span>
                <select
                  aria-label="Replay speed"
                  value={speed}
                  disabled={running || busy}
                  onChange={(e) => setSpeed(Number(e.target.value))}
                >
                  {[0.25, 0.5, 1, 2, 4].map((n) => (
                    <option key={n} value={n}>
                      {n}×
                    </option>
                  ))}
                </select>
              </label>
            </>
          ) : (
            <>
              <fieldset className="field">
                <span>Live receivers to fuse</span>
                {receivers.map((b) => (
                  <label className="checkbox" key={b.identity}>
                    <input
                      type="checkbox"
                      aria-label={`Live receiver ${b.logical_name}`}
                      checked={liveReceivers.includes(b.identity)}
                      disabled={running || busy}
                      onChange={(e) =>
                        setLiveReceivers((old) =>
                          toggle(old, b.identity, e.target.checked),
                        )
                      }
                    />
                    <span>
                      {b.logical_name} ·{" "}
                      {ports.find((p) => p.identity === b.identity)?.port}
                    </span>
                  </label>
                ))}
                {!receivers.length && (
                  <span className="subtle">No connected CSI receivers</span>
                )}
              </fieldset>
              <label className="field">
                <span>Serial baud rate</span>
                <input
                  aria-label="Serial baud rate"
                  type="number"
                  min={9600}
                  max={3000000}
                  value={baud}
                  disabled={running || busy}
                  onChange={(e) => setBaud(Number(e.target.value))}
                />
              </label>
            </>
          )}
          {source === "live" && (
            <label className="field">
              <span>Live camera (optional)</span>
              <select
                aria-label="Live camera (optional)"
                value={cameraDevice}
                disabled={running || busy}
                onChange={(e) => setCameraDevice(e.target.value)}
              >
                <option value="">No camera</option>
                {cameras.map((c) => (
                  <option key={c.device} value={c.device}>
                    {c.name || c.device}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
        {selectedModel && (
          <p className="subtle">
            {selectedModel.preprocessing.window_seconds}s window ·{" "}
            {selectedModel.preprocessing.sample_rate_hz} Hz · 52 subcarriers ·
            Classes: {selectedModel.classes.join(", ")} · Fusing{" "}
            {chosen.length} receiver{chosen.length === 1 ? "" : "s"}
          </p>
        )}
        {source === "replay" && (
          <p className="subtle">
            Replay preserves recorded CSI timing, and every selected receiver is
            replayed on the recording’s own clock so their windows stay aligned.
            Using the model’s own training session is a demonstration, not a
            measure of accuracy on new recordings. Recorded video follows the
            same capture timestamps and replay speed as CSI.
          </p>
        )}
        {cameraDevice && (
          <p className="subtle">
            USB camera settings follow Hardware Setup; phone settings follow the
            paired phone link.
          </p>
        )}
        <div className="actions">
          <button
            className="primary"
            disabled={
              busy ||
              running ||
              !!serviceError ||
              !selectedModel ||
              !chosen.length
            }
            onClick={() => void start()}
          >
            Start deployment
          </button>
          <button disabled={busy || !running} onClick={() => void stop()}>
            Stop deployment
          </button>
          <span role="status">{state.status}</span>
        </div>
        <p className="subtle">
          Deployment continues when you leave this page. Stop it before
          recording or training.
        </p>
        {catalog.errors.map((e: Json) => (
          <p className="notice warning" key={e.session_id}>
            {e.session_id}: {e.error}
          </p>
        ))}
      </section>
      <section className="panel">
        <div className="panel-heading">
          <h2>
            {state.status === "completed"
              ? "Final replay prediction"
              : "Current inferred pose"}
          </h2>
          <span className="subtle">Fused CSI model output</span>
        </div>
        {state.error && (
          <div className="alert" role="alert">
            {state.error}
          </div>
        )}
        {prediction ? (
          <>
            <div className="deploy-prediction">
              <strong>{prediction.label}</strong>
              <span>
                {(prediction.confidence * 100).toFixed(1)}% fused model score
              </span>
            </div>
            <p className="subtle">
              Window ends at {prediction.source_elapsed_s.toFixed(1)}s · Fused{" "}
              {prediction.fused_receivers.length} of{" "}
              {(state.receiver_names || prediction.fused_receivers).length}{" "}
              receivers ({prediction.fused_receivers.join(", ")}) · Inference{" "}
              {prediction.inference_ms.toFixed(1)} ms
            </p>
            {!!prediction.uncovered_receivers?.length && (
              <p className="notice warning">
                Not enough CSI coverage in this window for:{" "}
                {prediction.uncovered_receivers.join(", ")}. The pose is fused
                from the remaining receivers.
              </p>
            )}
            <div className="deploy-scores">
              {Object.entries(prediction.scores).map(([label, score]) => (
                <label key={label}>
                  <span>{label}</span>
                  <meter min={0} max={1} value={score as number} />
                  <span>{((score as number) * 100).toFixed(1)}%</span>
                </label>
              ))}
            </div>
            {prediction.receivers?.length > 1 && (
              <details>
                <summary>Per-receiver scores before fusion</summary>
                <table>
                  <thead>
                    <tr>
                      <th>Receiver</th>
                      <th>Action</th>
                      <th>Model score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {prediction.receivers.map((r: Json) => (
                      <tr key={r.receiver}>
                        <td>{r.receiver}</td>
                        <td>{r.label}</td>
                        <td>{(r.confidence * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </details>
            )}
          </>
        ) : (
          <p>
            {serviceError
              ? "Live prediction unavailable"
              : running
                ? signalText[state.signal] || "Loading model…"
                : "Start deployment to see a prediction."}
          </p>
        )}
        <p className="subtle">
          Accepted packets: {state.accepted || 0} · Rejected:{" "}
          {Object.values(state.rejected || {}).reduce(
            (a: number, b) => a + Number(b),
            0,
          )}{" "}
          · Transport dropped: {state.transport_dropped || 0}
        </p>
        {!!state.receiver_stats?.length && (
          <p className="subtle">
            {state.receiver_stats
              .map(
                (r: Json) =>
                  `${r.receiver}: ${r.accepted} accepted${r.live ? "" : " · no fresh data"}`,
              )
              .join(" · ")}
          </p>
        )}
        {Object.keys(state.rejected || {}).length > 0 && (
          <p className="subtle">
            {Object.entries(state.rejected)
              .map(([key, count]) => `${key.replaceAll("_", " ")}: ${count}`)
              .join(" · ")}
          </p>
        )}
        {Object.keys(state.receiver_errors || {}).length > 0 && (
          <p className="notice warning">
            {Object.entries(state.receiver_errors)
              .map(([name, message]) => `${name}: ${message}`)
              .join(" · ")}
          </p>
        )}
        {state.options?.source === "replay" && (
          <>
            <h3>Recorded video · synchronized with CSI replay</h3>
            {(state.video_error || videoError) && (
              <p className="notice warning">
                Video: {state.video_error || videoError}. CSI inference
                continues.
              </p>
            )}
            {state.video_status === "preparing" && (
              <p className="subtle">
                Indexing the recorded video for playback…
              </p>
            )}
            {state.video_available && (
              <>
                <video
                  key={`${state.id}:${state.video_path}`}
                  ref={video}
                  aria-label="Synchronized replay video"
                  muted
                  playsInline
                  preload="auto"
                  src={`/api/sessions/${encodeURIComponent(state.options.replay_session_id)}/files/${state.video_path || "raw/video.mp4"}`}
                  onPlaying={() => setVideoError("")}
                  onError={() =>
                    setVideoError(
                      "Recorded video is unavailable or cannot be decoded",
                    )
                  }
                />
                <p className="subtle" role="status">
                  Video offset from the CSI playhead:{" "}
                  {(drift.current * 1000).toFixed(0)} ms
                </p>
              </>
            )}
          </>
        )}
        {liveCamera && (
          <>
            <h3>Live camera · aligned with the fused CSI clock</h3>
            {state.camera_error && (
              <p className="notice warning">
                Camera: {state.camera_error}. CSI inference continues.
              </p>
            )}
            {!preview && (
              <p className="subtle">
                Waiting for a camera frame captured at the current CSI time…
              </p>
            )}
            <div className="preview">
              {preview && (
                <img src={preview.url} alt="Deployment live camera" />
              )}
            </div>
            {preview && (
              <p className="subtle" role="status">
                Camera frame offset from the fused CSI clock:{" "}
                {preview.skew === null
                  ? "unknown"
                  : `${preview.skew.toFixed(0)} ms`}
              </p>
            )}
          </>
        )}
        {!!state.history?.length && (
          <details>
            <summary>Recent predictions</summary>
            <table>
              <thead>
                <tr>
                  <th>Source time</th>
                  <th>Action</th>
                  <th>Fused score</th>
                  <th>Receivers</th>
                </tr>
              </thead>
              <tbody>
                {[...state.history].reverse().map((p: Json, i: number) => (
                  <tr key={i}>
                    <td>{p.source_elapsed_s.toFixed(1)}s</td>
                    <td>{p.label}</td>
                    <td>{(p.confidence * 100).toFixed(1)}%</td>
                    <td>{p.fused_receivers?.length ?? 1}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        )}
      </section>
    </>
  );
}
