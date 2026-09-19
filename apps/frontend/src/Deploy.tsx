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
  const [replayReceiver, setReplayReceiver] = useState("");
  const [receiver, setReceiver] = useState("");
  const [speed, setSpeed] = useState(1);
  const [baud, setBaud] = useState(921600);
  const [cameraDevice, setCameraDevice] = useState("");
  const [error, setError] = useState("");
  const [serviceError, setServiceError] = useState("");
  const [busy, setBusy] = useState(false);
  const [tick, setTick] = useState(0);
  const [previewReady, setPreviewReady] = useState(false);
  const initialized = useRef(false);
  const video = useRef<HTMLVideoElement>(null);
  const [videoError, setVideoError] = useState("");
  useEffect(() => {
    const player = video.current;
    if (!player) return;
    const sync = () => {
      const position = state.video_time_s;
      player.playbackRate = state.options?.replay_speed || 1;
      const playing =
        state.status === "running" &&
        state.video_playing !== false &&
        !serviceError;
      if (
        Number.isFinite(position) &&
        Math.abs(player.currentTime - position) > (playing ? 0.25 : 0.02)
      )
        player.currentTime = position;
      if (playing) {
        void player.play().catch((error: DOMException) => {
          if (error.name !== "AbortError")
            setVideoError(
              "Video playback could not start. Check browser autoplay settings.",
            );
        });
      } else {
        player.pause();
      }
    };
    sync();
    player.addEventListener("loadedmetadata", sync);
    return () => player.removeEventListener("loadedmetadata", sync);
  }, [state, serviceError]);
  const running = active(state.status);
  const receivers = boards.filter(
    (b) =>
      b.role === "csi_receiver" && ports.some((p) => p.identity === b.identity),
  );
  const selectedModel = catalog.models.find(
    (m: Json) => m.session_id === model,
  );
  const replayReceivers =
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
  useEffect(() => {
    if (running) return;
    setReplayReceiver((old) =>
      replayReceivers.includes(old) ? old : replayReceivers[0] || "",
    );
  }, [replay, catalog, running]);
  useEffect(() => {
    setReceiver((old) => old || receivers[0]?.identity || "");
  }, [boards, ports]);
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const value = await api("/status");
        if (!alive) return;
        setState(value);
        setTick((t) => t + 1);
        setServiceError("");
        if (!initialized.current && active(value.status) && value.options) {
          const o = value.options;
          setModel(o.model_session_id);
          setSource(o.source);
          setReplay(o.replay_session_id || "");
          setReplayReceiver(o.replay_receiver || "");
          setReceiver(o.receiver?.identity || "");
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
  async function start() {
    setBusy(true);
    setError("");
    setPreviewReady(false);
    setVideoError("");
    try {
      const b = receivers.find((x) => x.identity === receiver);
      const p = ports.find((x) => x.identity === receiver);
      if (source === "live" && (!b || !p))
        throw new Error("Receiver disconnected; refresh hardware.");
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
        receiver:
          source === "live" && b && p
            ? {
                identity: b.identity,
                logical_name: b.logical_name,
                port: p.port,
              }
            : null,
        replay_session_id: source === "replay" ? replay : null,
        replay_receiver: source === "replay" ? replayReceiver : null,
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
          Predict actions from one CSI receiver. Live capture and recorded
          replay use the selected model’s preprocessing. Camera images are for
          visualization only.
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
              <label className="field">
                <span>Replay receiver</span>
                <select
                  aria-label="Replay receiver"
                  value={replayReceiver}
                  disabled={running || busy}
                  onChange={(e) => setReplayReceiver(e.target.value)}
                >
                  {replayReceivers.map((name: string) => (
                    <option key={name}>{name}</option>
                  ))}
                </select>
              </label>
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
              <label className="field">
                <span>Live receiver</span>
                <select
                  aria-label="Live receiver"
                  value={receiver}
                  disabled={running || busy}
                  onChange={(e) => setReceiver(e.target.value)}
                >
                  <option value="">Choose a connected receiver</option>
                  {receivers.map((b) => (
                    <option key={b.identity} value={b.identity}>
                      {b.logical_name} ·{" "}
                      {ports.find((p) => p.identity === b.identity)?.port}
                    </option>
                  ))}
                </select>
              </label>
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
            Classes: {selectedModel.classes.join(", ")}
          </p>
        )}
        {source === "replay" && (
          <p className="subtle">
            Replay preserves recorded CSI timing. Using the model’s own training
            session is a demonstration, not a measure of accuracy on new
            recordings. Recorded video follows the same capture timestamps and
            replay speed as CSI.
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
              (source === "replay"
                ? !replayReceiver
                : !receivers.some((b) => b.identity === receiver))
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
          <span className="subtle">CSI model output</span>
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
                {(prediction.confidence * 100).toFixed(1)}% model score
              </span>
            </div>
            <p className="subtle">
              Window ends at {prediction.source_elapsed_s.toFixed(1)}s ·
              Inference {prediction.inference_ms.toFixed(1)} ms
            </p>
            <div className="deploy-scores">
              {Object.entries(prediction.scores).map(([label, score]) => (
                <label key={label}>
                  <span>{label}</span>
                  <meter min={0} max={1} value={score as number} />
                  <span>{((score as number) * 100).toFixed(1)}%</span>
                </label>
              ))}
            </div>
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
        {Object.keys(state.rejected || {}).length > 0 && (
          <p className="subtle">
            {Object.entries(state.rejected)
              .map(([key, count]) => `${key.replaceAll("_", " ")}: ${count}`)
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
            {state.video_available && (
              <video
                key={state.id || state.options.replay_session_id}
                ref={video}
                aria-label="Synchronized replay video"
                muted
                playsInline
                preload="auto"
                src={`/api/sessions/${encodeURIComponent(state.options.replay_session_id)}/files/raw/video.mp4`}
                onPlaying={() => setVideoError("")}
                onError={() =>
                  setVideoError(
                    "Recorded video is unavailable or cannot be decoded",
                  )
                }
              />
            )}
          </>
        )}
        {state.options?.source === "live" &&
          state.options?.camera &&
          state.capture_id && (
            <>
              <h3>Live camera · aligned with CSI capture time</h3>
              {state.camera_error && (
                <p className="notice warning">
                  Camera: {state.camera_error}. CSI inference continues.
                </p>
              )}
              {!previewReady && (
                <p className="subtle">Waiting for a fresh camera frame…</p>
              )}
              <div className="preview">
                <img
                  src={`/api/deploy/camera/${state.capture_id}?frame=${tick}${state.source_timestamp_ns ? `&timestamp_ns=${state.source_timestamp_ns}` : ""}`}
                  alt="Deployment live camera"
                  onLoad={() => setPreviewReady(true)}
                  onError={() => setPreviewReady(false)}
                  style={{ visibility: previewReady ? "visible" : "hidden" }}
                />
              </div>
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
                  <th>Model score</th>
                </tr>
              </thead>
              <tbody>
                {[...state.history].reverse().map((p: Json, i: number) => (
                  <tr key={i}>
                    <td>{p.source_elapsed_s.toFixed(1)}s</td>
                    <td>{p.label}</td>
                    <td>{(p.confidence * 100).toFixed(1)}%</td>
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
