import React, { useEffect, useState } from "react";
import { readApiResponse } from "./api";

type Json = Record<string, any>;
async function api(path: string, body?: Json) {
  const response = await fetch(
    `/api/train${path}`,
    body === undefined
      ? undefined
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  return readApiResponse(response);
}

export function Train({ sessions }: { sessions: Json[] }) {
  const [sid, setSid] = useState("");
  const [health, setHealth] = useState<Json | null>(null);
  const [state, setState] = useState<Json | null>(null);
  const [error, setError] = useState("");
  const [serviceError, setServiceError] = useState("");
  const [busy, setBusy] = useState(false);
  const [logs, setLogs] = useState("");
  const [selectedJob, setSelectedJob] = useState("");
  const [refresh, setRefresh] = useState(0);
  const [options, setOptions] = useState({
    epochs_frozen: 5,
    epochs_finetune: 15,
    batch_size: 8,
    window_seconds: 2,
  });
  useEffect(() => {
    if (!sid && sessions.length)
      setSid(
        (sessions.find((s) => s.status === "complete") || sessions[0])
          .session_id,
      );
  }, [sessions, sid]);
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    setState(null);
    setLogs("");
    async function poll() {
      try {
        const h = await api("/health");
        const data = sid
          ? await api(`/sessions/${encodeURIComponent(sid)}`)
          : null;
        const job =
          data?.jobs?.find((j: Json) => j.id === selectedJob) ||
          data?.jobs?.[0];
        const text = job ? await api(`/jobs/${job.id}/logs`) : { logs: "" };
        if (alive) {
          setHealth(h);
          setState(data);
          setLogs(text.logs);
          setServiceError("");
        }
      } catch (e) {
        if (alive) {
          setHealth(null);
          setServiceError((e as Error).message);
        }
      } finally {
        if (alive) timer = setTimeout(poll, 2000);
      }
    }
    void poll();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [sid, selectedJob, refresh]);
  const session = sessions.find((s) => s.session_id === sid);
  const latest = state?.jobs?.[0];
  const running =
    !!health?.active_job ||
    state?.jobs?.some((j: Json) => ["queued", "running"].includes(j.status));
  const disabled =
    busy ||
    running ||
    !health?.gpu?.ready ||
    session?.status !== "complete" ||
    !state;
  async function start(action: string) {
    if (!sid) return;
    setBusy(true);
    setError("");
    try {
      const job = await api(`/sessions/${encodeURIComponent(sid)}/start`, {
        action,
        ...options,
      });
      setSelectedJob(job.id);
      setRefresh((x) => x + 1);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function cancel() {
    const job = state?.jobs?.find((j: Json) =>
      ["queued", "running"].includes(j.status),
    );
    if (!job) return;
    setBusy(true);
    setError("");
    try {
      await api(`/jobs/${job.id}/cancel`, {});
      setRefresh((x) => x + 1);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const fileUrl = (path: string) =>
    `/api/sessions/${encodeURIComponent(sid)}/files/${path.split("/").map(encodeURIComponent).join("/")}`;
  return (
    <>
      {serviceError && (
        <div className="notice warning" role="status">
          {serviceError}
        </div>
      )}
      {error && (
        <div className="alert" role="alert">
          {error}
        </div>
      )}
      <section className="panel">
        <div className="panel-heading">
          <h2>Train from a recording</h2>
          <span className="subtle">
            {health?.gpu?.ready
              ? `${health.gpu.device} · ROCm ${health.gpu.rocm_sdk || health.gpu.rocm}`
              : "ROCm GPU required"}
          </span>
        </div>
        <label className="field">
          <span>Training session</span>
          <select
            aria-label="Training session"
            value={sid}
            disabled={busy}
            onChange={(e) => {
              setSid(e.target.value);
              setSelectedJob("");
              setError("");
            }}
          >
            <option value="">Choose a session</option>
            {sessions.map((s) => (
              <option key={s.session_id} value={s.session_id}>
                {s.session_id} · {s.status}
              </option>
            ))}
          </select>
        </label>
        {session && session.status !== "complete" && (
          <p className="notice warning">
            Choose a completed recording with its video, frame index and CSI
            files.
          </p>
        )}
        {health && !health.gpu?.ready && (
          <p className="notice warning">{health.gpu?.error}</p>
        )}
        <p>
          Label the session video, then fine-tune a copy of the pretrained CSI
          model. Auto train runs labeling, preprocessing and fine-tuning in
          order.
        </p>
        <div className="actions">
          <button disabled={disabled} onClick={() => void start("label")}>
            Labeling
          </button>
          <button
            disabled={disabled || !state?.labels_ready}
            onClick={() => void start("finetune")}
          >
            Fine-tune
          </button>
          <button
            className="primary"
            disabled={disabled}
            onClick={() => void start("auto")}
          >
            Auto train
          </button>
          {state?.jobs?.some((j: Json) =>
            ["queued", "running"].includes(j.status),
          ) && (
            <button
              className="danger"
              disabled={busy}
              onClick={() => void cancel()}
            >
              Cancel training
            </button>
          )}
        </div>
        <p className="hint">
          Automatic labels: Static, Walking, Sitting, Standing and Falling.
          Record one clearly visible person and at least two actions. Review the
          generated CSV before fine-tuning when possible. Every run starts from
          the original pretrained backbone and is saved separately.
        </p>
        <details>
          <summary>Fine-tuning settings</summary>
          <div className="form-grid">
            {(
              [
                ["epochs_frozen", "Head training epochs", 0, 200, 1],
                ["epochs_finetune", "Fine-tuning epochs", 0, 200, 1],
                ["batch_size", "Batch size", 1, 64, 1],
                ["window_seconds", "CSI window (seconds)", 0.5, 10, 0.5],
              ] as const
            ).map(([key, label, min, max, step]) => (
              <label className="field" key={key}>
                <span>{label}</span>
                <input
                  aria-label={label}
                  type="number"
                  min={min}
                  max={max}
                  step={step}
                  disabled={busy || running}
                  value={options[key]}
                  onChange={(e) =>
                    setOptions((o) => ({ ...o, [key]: Number(e.target.value) }))
                  }
                />
              </label>
            ))}
          </div>
          <p className="hint">
            CSI is resampled to 100 Hz using recorded capture times. Windows
            stay inside one labeled action; train and validation sets keep each
            action interval and all its receivers together.
          </p>
        </details>
      </section>
      {state && (
        <section className="panel">
          <div className="panel-heading">
            <h2>Results</h2>
            <span role="status">
              {latest
                ? `${latest.status} · ${latest.progress?.stage || latest.action}`
                : "No training runs yet"}
            </span>
          </div>
          {latest?.error && (
            <pre className="log-output" role="alert">
              {latest.error}
            </pre>
          )}
          <div className="actions">
            {state.labels_ready && (
              <a href={fileUrl("train/action_results.csv")} download>
                Download action_results.csv
              </a>
            )}
            {state.model && (
              <>
                <a href={fileUrl(state.model.model_path)} download>
                  Download fine-tuned model
                </a>
                <a
                  href={fileUrl(
                    `train/runs/${state.model.run_id}/classes.json`,
                  )}
                  download
                >
                  Download class mapping
                </a>
                <a
                  href={fileUrl(`train/runs/${state.model.run_id}/model.json`)}
                  download
                >
                  Download training metadata
                </a>
              </>
            )}
          </div>
          {state.model && (
            <p>
              {state.model.preprocessing.windows} CSI windows ·{" "}
              {state.model.metrics.validation_available
                ? `Best validation macro-F1: ${state.model.metrics.best_validation_macro_f1.toFixed(3)}`
                : "No independent validation intervals; model trained without a validation score."}
            </p>
          )}
          {state.model?.metrics.validation_missing_classes?.length > 0 && (
            <p className="hint">
              No validation intervals for:{" "}
              {state.model.metrics.validation_missing_classes.join(", ")}.
            </p>
          )}
          {state.model_stale && (
            <p className="notice warning">
              The current labels differ from the labels used by this model.
              Fine-tune again to apply them.
            </p>
          )}
          <p className="hint">
            Outputs are saved in data/sessions/{sid}/train/. Raw recordings are
            preserved. Run history retains earlier models and label files.
          </p>
          {state.jobs?.length > 0 && (
            <label className="field">
              <span>Training run logs</span>
              <select
                aria-label="Training run logs"
                value={selectedJob || latest.id}
                onChange={(e) => setSelectedJob(e.target.value)}
              >
                {state.jobs.map((j: Json) => (
                  <option value={j.id} key={j.id}>
                    {j.created_at} · {j.action} · {j.status}
                  </option>
                ))}
              </select>
            </label>
          )}
          <pre className="log-output" aria-label="Training logs">
            {logs || "Job progress and logs will appear here."}
          </pre>
        </section>
      )}
    </>
  );
}
