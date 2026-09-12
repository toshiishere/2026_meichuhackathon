import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

type Json = Record<string, any>;
type Board = {
  identity: string;
  logical_name: string;
  role: string;
  port: string;
  target: string;
  geometry: string;
  firmware: Json;
};
type Camera = {
  device: string;
  width: number;
  height: number;
  fps: number;
  geometry: string;
};
const api = async (path: string, body?: unknown) => {
  const response = await fetch(
    "/api" + path,
    body === undefined
      ? undefined
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail),
    );
  return data;
};
const number = (v: unknown, places = 1) =>
  typeof v === "number" ? v.toFixed(places) : "—";
const bytes = (v: number) =>
  v >= 1e9
    ? `${(v / 1e9).toFixed(2)} GB`
    : v >= 1e6
      ? `${(v / 1e6).toFixed(1)} MB`
      : v >= 1000
        ? `${(v / 1000).toFixed(1)} kB`
        : `${v || 0} B`;
const age = (v: number) =>
  `${Math.floor((v || 0) / 60)
    .toString()
    .padStart(2, "0")}:${Math.floor((v || 0) % 60)
    .toString()
    .padStart(2, "0")}`;
function Badge({ value }: { value: string }) {
  return (
    <span className={`badge ${value}`}>
      {value?.replaceAll("_", " ") || "unknown"}
    </span>
  );
}
function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  const id = React.useId();
  return (
    <label className="field" htmlFor={id}>
      <span id={`${id}-label`}>{label}</span>
      {React.cloneElement(
        children as React.ReactElement<{
          id?: string;
          "aria-labelledby"?: string;
        }>,
        {
          id,
          "aria-labelledby": `${id}-label`,
        },
      )}
    </label>
  );
}
function Metric({
  label,
  value,
  sub,
}: {
  label: string;
  value: React.ReactNode;
  sub?: string;
}) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
      {sub && <small>{sub}</small>}
    </div>
  );
}
function Empty({ children }: { children: React.ReactNode }) {
  return <div className="empty">{children}</div>;
}

function App() {
  const [page, setPage] = useState("Dashboard");
  const [health, setHealth] = useState<Json>({});
  const defaultsLoaded = React.useRef(false);
  const [connected, setConnected] = useState(false);
  const [ports, setPorts] = useState<Json[]>([]);
  const [boards, setBoards] = useState<Board[]>([]);
  const [cameras, setCameras] = useState<Json[]>([]);
  const [sessions, setSessions] = useState<Json[]>([]);
  const [jobs, setJobs] = useState<Json[]>([]);
  const [status, setStatus] = useState<Json>({ status: "idle" });
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [selectedPort, setSelectedPort] = useState("");
  const [assignment, setAssignment] = useState({
    logical_name: "rx_left",
    role: "csi_receiver",
    target: "esp32c3",
    geometry: "",
  });
  const [gpio, setGpio] = useState("8");
  const [ledType, setLedType] = useState("gpio");
  const [activeLow, setActiveLow] = useState(false);
  const [camera, setCamera] = useState<Camera>({
    device: "",
    width: 1280,
    height: 720,
    fps: 30,
    geometry: "",
  });
  const [preview, setPreview] = useState("");
  const previewToken = React.useRef(crypto.randomUUID());
  const closePreview = () => {
    setPreview("");
    api("/hardware/camera/close", { token: previewToken.current }).catch((e) =>
      setError(e.message),
    );
  };
  const [sender, setSender] = useState("");
  const [receivers, setReceivers] = useState<string[]>([]);
  const [form, setForm] = useState({
    session_id: `session_${new Date()
      .toISOString()
      .replace(/[-:TZ.]/g, "")
      .slice(0, 14)}`,
    subject_id: "",
    room_id: "",
    layout_id: "",
    description: "",
    activity_script: "",
    notes: "",
    layout_notes: "",
    baud_rate: 921600,
    expected_rate_hz: 100,
    min_rate_ratio: 0.7,
    duration_seconds: "",
    synthetic_loss: 0.01,
    synthetic_seed: 42,
  });
  const [selectedJob, setSelectedJob] = useState("");
  const [jobLogs, setJobLogs] = useState("");
  const [detail, setDetail] = useState<Json | null>(null);
  const [preflight, setPreflight] = useState<Json | null>(null);
  const [flashChoice, setFlashChoice] = useState("csi-recv");
  const [confirmFlash, setConfirmFlash] = useState(false);
  const recording = ["starting", "preflight", "recording", "stopping"].includes(
    status.status,
  );
  const hardwareBusy = jobs.some((j) =>
    ["queued", "running"].includes(j.status),
  );
  const currentJob = jobs.find((j) => j.id === selectedJob);
  const mode = health.hardware?.mode || status.mode || "connecting";

  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(false);
    }
  };
  const refresh = async () => {
    const [p, b, c, s, h] = await Promise.all([
      api("/hardware/serial"),
      api("/devices"),
      api("/hardware/cameras"),
      api("/sessions"),
      api("/health"),
    ]);
    setPorts(p);
    setBoards(b);
    setCameras(c);
    setSessions(s);
    setHealth(h);
    if (!defaultsLoaded.current && h.hardware?.defaults) {
      const d = h.hardware.defaults;
      defaultsLoaded.current = true;
      setForm((old) => ({...old, baud_rate: d.baud_rate,
        expected_rate_hz: d.expected_rate_hz, min_rate_ratio: d.min_rate_ratio,
        synthetic_loss: d.synthetic_loss, synthetic_seed: d.synthetic_seed}));
      setCamera((old) => ({...old, width: d.camera_width,
        height: d.camera_height, fps: d.camera_fps}));
    }
    setSelectedPort((old) => old || p[0]?.port || "");
    setCamera((old) => ({ ...old, device: old.device || c[0]?.device || "" }));
    setSender(
      (old) =>
        old || b.find((x: Board) => x.role === "csi_sender")?.identity || "",
    );
    setReceivers((old) =>
      old.length
        ? old
        : b
            .filter((x: Board) => x.role === "csi_receiver")
            .map((x: Board) => x.identity),
    );
  };
  useEffect(() => {
    run(refresh);
    const events = new EventSource("/api/events");
    events.onmessage = (e) => {
      const d = JSON.parse(e.data);
      if (d.error) {
        setConnected(false);
        return;
      }
      setConnected(true);
      setJobs(d.jobs);
      setStatus(d.collection);
    };
    events.onerror = () => setConnected(false);
    return () => events.close();
  }, []);
  useEffect(() => {
    if (!selectedJob) return;
    api(`/jobs/${selectedJob}/logs`)
      .then((d) => setJobLogs(d.logs))
      .catch((e) => setError(e.message));
  }, [selectedJob, jobs]);
  useEffect(() => {
    if (currentJob?.status === "completed" && currentJob.kind === "preflight")
      setPreflight(currentJob.result);
  }, [currentJob?.status, selectedJob]);
  useEffect(() => {
    if (["complete", "incomplete"].includes(status.status))
      api("/sessions")
        .then(setSessions)
        .catch((e) => setError(e.message));
  }, [status.status]);
  useEffect(() => {
    const b = boards.find(
      (x) =>
        x.port === selectedPort ||
        x.identity === ports.find((p) => p.port === selectedPort)?.identity,
    );
    if (b)
      setAssignment({
        logical_name: b.logical_name,
        role: b.role,
        target: b.target,
        geometry: b.geometry,
      });
  }, [selectedPort, boards]);
  const submitJob = async (path: string, body: unknown) => {
    const j = await api(path, body);
    setSelectedJob(j.id);
    setJobs((old) => [j, ...old.filter((x) => x.id !== j.id)]);
  };
  const choosePage = (value: string) => {
    closePreview();
    setPage(value);
  };
  const config = () => {
    const s = boards.find((b) => b.identity === sender);
    if (!s)
      throw new Error(
        "Assign and select a CSI sender in Hardware Setup first.",
      );
    const selected = boards.filter(
      (b) => receivers.includes(b.identity) && b.role === "csi_receiver",
    );
    if (!selected.length)
      throw new Error("Assign and select at least one CSI receiver.");
    if (!camera.device) throw new Error("Select a camera in Hardware Setup.");
    const toReceiver = (b: Board) => ({
      logical_name: b.logical_name,
      port: ports.find((p) => p.identity === b.identity)?.port || b.port,
      identity: b.identity,
      geometry: b.geometry,
      firmware: b.firmware,
    });
    return {
      ...form,
      duration_seconds: form.duration_seconds
        ? Number(form.duration_seconds)
        : null,
      sender: toReceiver(s),
      receivers: selected.map(toReceiver),
      camera,
    };
  };
  const assign = async () => {
    const p = ports.find((p) => p.port === selectedPort);
    if (!p) throw new Error("Select a connected device.");
    await api("/devices", {
      identity: p.identity,
      port: p.port,
      ...assignment,
      firmware: {},
    });
    await refresh();
  };
  const inspectSession = async (sid: string) => {
    setDetail(await api(`/sessions/${sid}`));
    choosePage("Sessions");
  };
  const cameraQuery = `token=${previewToken.current}&device=${encodeURIComponent(camera.device)}&width=${camera.width}&height=${camera.height}&fps=${camera.fps}`;
  const changeForm = (key: string, value: string | number) => {
    setForm((old) => ({ ...old, [key]: value }));
    setPreflight(null);
  };

  return (
    <div className="shell">
      <aside>
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            choosePage("Dashboard");
          }}
        >
          <span className="brand-mark">≋</span>
          <span>
            CSI<span className="brand-light"> / LAB</span>
            <small>COLLECTION WORKSPACE</small>
          </span>
        </a>
        <div className="nav-label">WORKSPACE</div>
        <nav>
          {[
            "Dashboard",
            "Hardware Setup",
            "Data Collection",
            "Sessions",
            "System / Logs",
          ].map((p, i) => (
            <button
              key={p}
              className={page === p ? "selected" : ""}
              onClick={() => choosePage(p)}
            >
              <span className="nav-icon" aria-hidden="true">
                {["◫", "⌘", "◉", "▤", "≡"][i]}
              </span>
              {p}
              {p === "Data Collection" && recording && (
                <i className="live-dot" />
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <span className={`connection ${connected ? "on" : ""}`} />
          {connected ? "Services connected" : "Reconnecting…"}
          <small>Local research workspace</small>
          <Badge value={mode} />
        </div>
      </aside>
      <main>
        <header>
          <div className="breadcrumb">
            Workspace <span>/</span> {page}
          </div>
          <span className="clock-note">
            HOST MONOTONIC CLOCK <i className="live-dot" />
          </span>
        </header>
        <div className="content">
          <div className="page-heading">
            <div>
              <div className="eyebrow">SYNCHRONIZED ACQUISITION</div>
              <h1>{page}</h1>
              <p>
                {
                  (
                    {
                      Dashboard:
                        "A clear view of your devices, recordings, and data quality.",
                      "Hardware Setup":
                        "Identify your boards and verify every stream before recording.",
                      "Data Collection":
                        "Capture CSI and camera frames on one shared host clock.",
                      Sessions:
                        "Inspect recordings and download their original artifacts.",
                      "System / Logs":
                        "Hardware operations, results, and diagnostic logs.",
                    } as Json
                  )[page]
                }
              </p>
            </div>
            <button
              className="secondary"
              disabled={busy}
              onClick={() => run(refresh)}
            >
              ↻ Refresh
            </button>
          </div>
          {mode === "synthetic" && (
            <div className="notice">
              <strong>Synthetic hardware</strong> Generated CSI and video for
              testing. Recordings are marked synthetic.
            </div>
          )}
          {!connected && (
            <div className="notice warning">
              Live connection unavailable. Recording may still be active;
              reconnect before starting another session.
            </div>
          )}
          {error && (
            <div role="alert" className="alert">
              {error}
              <button onClick={() => setError("")} aria-label="Dismiss error">
                ×
              </button>
            </div>
          )}
          {status.last_error && (
            <div className="alert">Collection: {status.last_error}</div>
          )}

          {page === "Dashboard" && (
            <>
              <div className="metrics">
                <Metric
                  label="Connected boards"
                  value={ports.length}
                  sub={`${boards.length} logical assignments`}
                />
                <Metric
                  label="Cameras"
                  value={cameras.length}
                  sub="Available capture devices"
                />
                <Metric
                  label="Recorded sessions"
                  value={sessions.length}
                  sub={`${sessions.filter((s) => s.status === "complete").length} finalized`}
                />
                <Metric
                  label="Collector"
                  value={<Badge value={status.status} />}
                  sub={
                    recording
                      ? age(status.elapsed_seconds)
                      : "Ready for your next session"
                  }
                />
              </div>
              <section className="hero">
                <div>
                  <span className="eyebrow">YOUR NEXT EXPERIMENT</span>
                  <h2>
                    Reliable recordings start
                    <br />
                    with a verified setup.
                  </h2>
                  <p>
                    Check packet arrival, camera timing, and storage.
                    <br />
                    Then record all streams together.
                  </p>
                  <button
                    className="primary"
                    onClick={() =>
                      choosePage(
                        boards.length ? "Data Collection" : "Hardware Setup",
                      )
                    }
                  >
                    {boards.length ? "Prepare a recording" : "Set up hardware"}{" "}
                    <span>→</span>
                  </button>
                </div>
                <div className="flow-visual">
                  <div className="flow-source">
                    CSI RECEIVERS<small>TX sequence + host timestamp</small>
                  </div>
                  <div className="flow-source">
                    USB CAMERA<small>Frame timestamp + video PTS</small>
                  </div>
                  <div className="flow-line" />
                  <div className="flow-destination">
                    ONE SESSION<small>Original data. Shared time.</small>
                  </div>
                </div>
              </section>
              <section className="panel">
                <div className="panel-heading">
                  <h2>Recent recordings</h2>
                  <button
                    className="text-button"
                    onClick={() => choosePage("Sessions")}
                  >
                    View all →
                  </button>
                </div>
                <SessionTable
                  sessions={sessions.slice(0, 5)}
                  inspect={(sid) => run(() => inspectSession(sid))}
                />
              </section>
            </>
          )}

          {page === "Hardware Setup" && (
            <>
              <section className="panel">
                <div className="panel-heading">
                  <h2>ESP32 boards</h2>
                  <span className="subtle">Stable USB identity preferred</span>
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Port / stable path</th>
                        <th>USB serial</th>
                        <th>Logical name</th>
                        <th>Role</th>
                        <th>Target</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ports.map((p) => {
                        const b = boards.find((b) => b.identity === p.identity);
                        return (
                          <tr
                            key={p.identity}
                            className={
                              selectedPort === p.port ? "highlight" : ""
                            }
                            onClick={() => setSelectedPort(p.port)}
                          >
                            <td>
                              <button
                                className="text-button"
                                onClick={() => setSelectedPort(p.port)}
                              >
                                {p.port}
                              </button>
                              <small>{p.description}</small>
                            </td>
                            <td>{p.serial_number || "Unavailable"}</td>
                            <td>{b?.logical_name || "Unassigned"}</td>
                            <td>{b?.role.replace("csi_", "") || "—"}</td>
                            <td>{b?.target || "Not assigned"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                {!ports.length && (
                  <Empty>
                    No serial devices found. Connect the boards and check Docker
                    device access.
                  </Empty>
                )}
              </section>
              <div className="two-column">
                <section className="panel">
                  <h2>Board configuration</h2>
                  <Field label="Selected serial port">
                    <select
                      value={selectedPort}
                      onChange={(e) => setSelectedPort(e.target.value)}
                    >
                      {!ports.length && <option value="">No devices</option>}
                      {ports.map((p) => (
                        <option key={p.port}>{p.port}</option>
                      ))}
                    </select>
                  </Field>
                  <div className="form-grid">
                    <Field label="Logical name">
                      <input
                        value={assignment.logical_name}
                        onChange={(e) =>
                          setAssignment({
                            ...assignment,
                            logical_name: e.target.value,
                          })
                        }
                      />
                    </Field>
                    <Field label="Role">
                      <select
                        value={assignment.role}
                        onChange={(e) =>
                          setAssignment({ ...assignment, role: e.target.value })
                        }
                      >
                        <option value="csi_receiver">CSI Receiver</option>
                        <option value="csi_sender">CSI Sender</option>
                      </select>
                    </Field>
                    <Field label="ESP target">
                      <select
                        value={assignment.target}
                        onChange={(e) =>
                          setAssignment({
                            ...assignment,
                            target: e.target.value,
                          })
                        }
                      >
                        {["esp32", "esp32c3", "esp32s3", "esp32c6"].map((t) => (
                          <option key={t}>{t}</option>
                        ))}
                      </select>
                    </Field>
                    <Field label="Position / orientation (meters)">
                      <input
                        value={assignment.geometry}
                        onChange={(e) =>
                          setAssignment({
                            ...assignment,
                            geometry: e.target.value,
                          })
                        }
                        placeholder="x: 0, y: 2, z: 1.2; facing east"
                      />
                    </Field>
                  </div>
                  <div className="actions">
                    <button
                      className="primary"
                      disabled={busy || !selectedPort}
                      onClick={() => run(assign)}
                    >
                      Save assignment
                    </button>
                    <button
                      disabled={
                        busy || hardwareBusy || !!preview || !selectedPort
                      }
                      onClick={() =>
                        run(() =>
                          submitJob("/hardware/serial/probe", {
                            port: selectedPort,
                          }),
                        )
                      }
                    >
                      Probe board
                    </button>
                    <button
                      disabled={
                        busy || hardwareBusy || !!preview || !selectedPort
                      }
                      onClick={() =>
                        run(() =>
                          submitJob("/hardware/serial/test", {
                            port: selectedPort,
                            baud_rate: form.baud_rate,
                            expected_rate_hz: form.expected_rate_hz,
                            min_rate_ratio: form.min_rate_ratio,
                            seconds: 5,
                          }),
                        )
                      }
                    >
                      CSI test / monitor
                    </button>
                  </div>
                  <p className="hint">
                    Probe uses esptool and may reset the board. The CSI test
                    captures 5 seconds and shows a serial sample in its result.
                  </p>
                </section>
                <section className="panel">
                  <h2>Firmware</h2>
                  <p className="hint">
                    Preserved ESP-CSI source · IDF 5.5.0 · 100 Hz · channel 11 ·
                    HT40
                  </p>
                  <Field label="Firmware project">
                    <select
                      value={flashChoice}
                      onChange={(e) => {
                        setFlashChoice(e.target.value);
                        setConfirmFlash(false);
                      }}
                    >
                      <option value="csi-recv">CSI Receiver</option>
                      <option value="csi-send">CSI Sender</option>
                      <option value="blink">Blink / Identify</option>
                    </select>
                  </Field>
                  {flashChoice === "blink" && (
                    <div className="form-grid">
                      <Field label="LED type">
                        <select
                          value={ledType}
                          onChange={(e) => setLedType(e.target.value)}
                        >
                          <option value="gpio">GPIO LED</option>
                          <option value="rgb">WS2812 RGB</option>
                          <option value="none">No usable LED</option>
                        </select>
                      </Field>
                      <Field label="LED GPIO">
                        <input
                          type="number"
                          min="0"
                          max="48"
                          value={gpio}
                          onChange={(e) => setGpio(e.target.value)}
                        />
                      </Field>
                      <label className="checkbox">
                        <input
                          type="checkbox"
                          checked={activeLow}
                          onChange={(e) => setActiveLow(e.target.checked)}
                        />
                        Active low
                      </label>
                    </div>
                  )}
                  <label className="checkbox">
                    <input
                      type="checkbox"
                      checked={confirmFlash}
                      onChange={(e) => setConfirmFlash(e.target.checked)}
                    />
                    Replace firmware on the selected board
                  </label>
                  <div className="actions">
                    {["flash", "build", "rebuild", "clean-build"].map(
                      (operation) => (
                        <button
                          className={operation === "flash" ? "primary" : ""}
                          key={operation}
                          disabled={
                            busy ||
                            hardwareBusy ||
                            !!preview ||
                            !selectedPort ||
                            mode === "synthetic" ||
                            (operation === "flash" && !confirmFlash)
                          }
                          onClick={() =>
                            run(() =>
                              submitJob("/hardware/flash", {
                                port: selectedPort,
                                target: assignment.target,
                                firmware: flashChoice,
                                operation,
                                gpio:
                                  flashChoice === "blink" ? Number(gpio) : null,
                                led_type: ledType,
                                active_low: activeLow,
                              }),
                            )
                          }
                        >
                          {
                            (
                              {
                                flash: "Flash firmware",
                                build: "Build",
                                rebuild: "Rebuild",
                                "clean-build": "Clean build",
                              } as Json
                            )[operation]
                          }
                        </button>
                      ),
                    )}
                  </div>
                  <p className="hint">
                    Choose the actual chip target and LED pin from your board’s
                    pinout. Blink replaces the current program; restore CSI
                    firmware after identification. Builds reuse a
                    content-addressed cache.
                  </p>
                </section>
              </div>
              <section className="panel">
                <div className="panel-heading">
                  <h2>Camera setup</h2>
                  <span className="subtle">USB / UVC capture</span>
                </div>
                <div className="two-column">
                  <div>
                    <Field label="Capture device">
                      <select
                        value={camera.device}
                        onChange={(e) => {
                          closePreview();
                          setCamera({ ...camera, device: e.target.value });
                        }}
                      >
                        {!cameras.length && (
                          <option value="">No cameras</option>
                        )}
                        {cameras.map((c) => (
                          <option key={c.device} value={c.device}>
                            {c.name} · {c.device}
                          </option>
                        ))}
                      </select>
                    </Field>
                    <div className="form-grid">
                      {(["width", "height", "fps"] as const).map((key) => (
                        <Field
                          key={key}
                          label={key === "fps" ? "Requested FPS" : key}
                        >
                          <input
                            type="number"
                            min="1"
                            value={camera[key]}
                            onChange={(e) => {
                              closePreview();
                              setCamera({
                                ...camera,
                                [key]: Number(e.target.value),
                              });
                            }}
                          />
                        </Field>
                      ))}
                    </div>
                    <Field label="Camera position / orientation">
                      <input
                        value={camera.geometry}
                        onChange={(e) =>
                          setCamera({ ...camera, geometry: e.target.value })
                        }
                      />
                    </Field>
                    <div className="actions">
                      <button
                        disabled={busy || hardwareBusy || !camera.device}
                        onClick={() => {
                          if (preview) closePreview();
                          else
                            setPreview(
                              `/api/hardware/camera/stream?${cameraQuery}`,
                            );
                        }}
                      >
                        {preview ? "Close preview" : "Live preview"}
                      </button>
                      <button
                        disabled={
                          busy || hardwareBusy || !!preview || !camera.device
                        }
                        onClick={() =>
                          setPreview(
                            `/api/hardware/camera/snapshot?${cameraQuery}&t=${Date.now()}`,
                          )
                        }
                      >
                        Snapshot
                      </button>
                      <button
                        className="primary"
                        disabled={
                          busy || hardwareBusy || !!preview || !camera.device
                        }
                        onClick={() =>
                          run(() => submitJob("/hardware/camera/test", camera))
                        }
                      >
                        Test camera
                      </button>
                    </div>
                    <details>
                      <summary>Supported formats</summary>
                      <pre>
                        {cameras.find((c) => c.device === camera.device)
                          ?.formats || "No device selected"}
                      </pre>
                    </details>
                  </div>
                  <div className="preview">
                    {preview ? (
                      <img
                        src={preview}
                        alt="Camera preview"
                        onError={() => {
                          closePreview();
                          setError(
                            "Camera preview failed. Check the camera and close any other camera clients.",
                          );
                        }}
                      />
                    ) : (
                      <div>
                        <span>◎</span>
                        <p>Camera preview</p>
                        <small>
                          Open a preview to check framing.
                          <br />
                          Close it before testing or recording.
                        </small>
                      </div>
                    )}
                  </div>
                </div>
              </section>
            </>
          )}

          {page === "Data Collection" && (
            <>
              <div className="collection-layout">
                <section className="panel">
                  <h2>Session details</h2>
                  <fieldset disabled={recording || hardwareBusy || busy}>
                    <div className="form-grid">
                      {["session_id", "subject_id", "room_id", "layout_id"].map(
                        (key) => (
                          <Field
                            key={key}
                            label={key
                              .replaceAll("_", " ")
                              .replace(/^./, (c) => c.toUpperCase())}
                          >
                            <input
                              value={(form as Json)[key]}
                              onChange={(e) => changeForm(key, e.target.value)}
                              required={key === "session_id"}
                            />
                          </Field>
                        ),
                      )}
                    </div>
                    {[
                      "description",
                      "activity_script",
                      "notes",
                      "layout_notes",
                    ].map((key) => (
                      <Field
                        key={key}
                        label={key
                          .replaceAll("_", " ")
                          .replace(/^./, (c) => c.toUpperCase())}
                      >
                        <textarea
                          rows={key === "notes" ? 3 : 2}
                          value={(form as Json)[key]}
                          onChange={(e) => changeForm(key, e.target.value)}
                          placeholder={
                            key === "activity_script"
                              ? "Describe the planned recording sequence"
                              : ""
                          }
                        />
                      </Field>
                    ))}
                    <h3>Hardware selection</h3>
                    <Field label="CSI sender">
                      <select
                        value={sender}
                        onChange={(e) => {
                          setSender(e.target.value);
                          setPreflight(null);
                        }}
                      >
                        <option value="">Select sender</option>
                        {boards
                          .filter((b) => b.role === "csi_sender")
                          .map((b) => (
                            <option key={b.identity} value={b.identity}>
                              {b.logical_name} · {b.port}
                            </option>
                          ))}
                      </select>
                    </Field>
                    <div className="field">
                      <span>CSI receivers</span>
                      {boards
                        .filter((b) => b.role === "csi_receiver")
                        .map((b) => (
                          <label
                            className="checkbox receiver-choice"
                            key={b.identity}
                          >
                            <input
                              type="checkbox"
                              checked={receivers.includes(b.identity)}
                              onChange={(e) => {
                                setReceivers(
                                  e.target.checked
                                    ? [...receivers, b.identity]
                                    : receivers.filter((x) => x !== b.identity),
                                );
                                setPreflight(null);
                              }}
                            />
                            <strong>{b.logical_name}</strong>
                            <small>{b.port}</small>
                          </label>
                        ))}
                      {!boards.some((b) => b.role === "csi_receiver") && (
                        <p className="hint">
                          Assign receiver boards in Hardware Setup first.
                        </p>
                      )}
                    </div>
                    <Field label="Camera">
                      <select
                        value={camera.device}
                        onChange={(e) => {
                          setCamera({ ...camera, device: e.target.value });
                          setPreflight(null);
                        }}
                      >
                        <option value="">Select camera</option>
                        {cameras.map((c) => (
                          <option key={c.device} value={c.device}>
                            {c.name}
                          </option>
                        ))}
                      </select>
                    </Field>
                    <h3>Capture settings</h3>
                    <div className="form-grid">
                      {[
                        "baud_rate",
                        "expected_rate_hz",
                        "min_rate_ratio",
                        "duration_seconds",
                      ].map((key) => (
                        <Field
                          key={key}
                          label={
                            (
                              {
                                baud_rate: "Serial baud rate",
                                expected_rate_hz: "Expected CSI rate (Hz)",
                                min_rate_ratio: "Minimum acceptable rate ratio",
                                duration_seconds:
                                  "Duration (seconds, optional)",
                              } as Json
                            )[key]
                          }
                        >
                          <input
                            type="number"
                            step={key === "min_rate_ratio" ? "0.05" : "1"}
                            value={(form as Json)[key]}
                            placeholder={
                              key === "duration_seconds" ? "Manual stop" : ""
                            }
                            onChange={(e) =>
                              changeForm(
                                key,
                                key === "duration_seconds"
                                  ? e.target.value
                                  : Number(e.target.value),
                              )
                            }
                          />
                        </Field>
                      ))}
                    </div>
                    <p className="hint">
                      Camera: {camera.width} × {camera.height} at {camera.fps}{" "}
                      FPS. Change resolution in Hardware Setup.
                    </p>
                    {mode === "synthetic" && (
                      <div className="form-grid">
                        <Field label="Synthetic packet loss (0–0.5)">
                          <input
                            type="number"
                            min="0"
                            max="0.5"
                            step="0.01"
                            value={form.synthetic_loss}
                            onChange={(e) =>
                              changeForm(
                                "synthetic_loss",
                                Number(e.target.value),
                              )
                            }
                          />
                        </Field>
                        <Field label="Synthetic random seed">
                          <input
                            type="number"
                            value={form.synthetic_seed}
                            onChange={(e) =>
                              changeForm(
                                "synthetic_seed",
                                Number(e.target.value),
                              )
                            }
                          />
                        </Field>
                      </div>
                    )}
                  </fieldset>
                </section>
                <div>
                  <section className="panel sticky">
                    <div className="panel-heading">
                      <h2>Recording control</h2>
                      <Badge value={status.status} />
                    </div>
                    <div className="timer">{age(status.elapsed_seconds)}</div>
                    <p className="center subtle">
                      {recording
                        ? status.session_id || "Checking hardware…"
                        : "Ready when your hardware is"}
                    </p>
                    <div className="record-actions">
                      <button
                        disabled={
                          busy || hardwareBusy || recording || !connected
                        }
                        onClick={() =>
                          run(() =>
                            submitJob("/collection/preflight", config()),
                          )
                        }
                      >
                        Run preflight
                      </button>
                      <button
                        className="primary"
                        disabled={
                          busy || hardwareBusy || recording || !connected
                        }
                        onClick={() =>
                          run(async () => {
                            closePreview();
                            await submitJob("/collection/start", config());
                          })
                        }
                      >
                        ● Start recording
                      </button>
                      <button
                        className="danger"
                        disabled={busy || !recording}
                        onClick={() => run(() => api("/collection/stop", {}))}
                      >
                        ■ Stop recording
                      </button>
                    </div>
                    <p className="hint">
                      Start always repeats preflight checks. A blank duration
                      allows manual stop. Finalization drains buffered data
                      before the session completes.
                    </p>
                    {(preflight || status.preflight) && (
                      <Preflight result={preflight || status.preflight} />
                    )}
                  </section>
                  <section className="panel">
                    <h3>Collection notes</h3>
                    <p className="hint">
                      Record board positions, antenna orientations, and room
                      layout for reproducibility. For simulated fall recordings,
                      use suitable padding and a controlled experiment.
                    </p>
                  </section>
                </div>
              </div>
              <LiveStatus status={status} />
            </>
          )}

          {page === "Sessions" && (
            <>
              <section className="panel">
                <div className="panel-heading">
                  <h2>Session archive</h2>
                  <button
                    onClick={() => run(() => api("/manifest/rebuild", {}))}
                  >
                    Rebuild manifest
                  </button>
                </div>
                <SessionTable
                  sessions={sessions}
                  inspect={(sid) => run(() => inspectSession(sid))}
                />
              </section>
              {detail && (
                <section className="panel">
                  <div className="panel-heading">
                    <div>
                      <span className="eyebrow">SESSION DETAIL</span>
                      <h2>{detail.session_id}</h2>
                    </div>
                    <Badge value={detail.status} />
                  </div>
                  {detail.status === "incomplete" && (
                    <div className="notice warning">
                      This recording did not finalize successfully. Original and
                      temporary artifacts are preserved for inspection.
                    </div>
                  )}
                  <div className="two-column">
                    <div>
                      {detail.artifacts?.some(
                        (f: Json) => f.path === "raw/video.mp4",
                      ) ? (
                        <video
                          controls
                          preload="metadata"
                          src={`/api/sessions/${detail.session_id}/files/raw/video.mp4`}
                        />
                      ) : (
                        <Empty>No finalized video available.</Empty>
                      )}
                      <p className="hint">
                        Video PTS starts at the first encoded frame. Use the
                        frame timestamp file to align playback with CSI on the
                        host clock.
                      </p>
                    </div>
                    <div className="artifact-list">
                      {detail.artifacts?.map((f: Json) => (
                        <a
                          key={f.path}
                          href={`/api/sessions/${detail.session_id}/files/${f.path}`}
                          target="_blank"
                          rel="noreferrer"
                        >
                          <span>{f.path}</span>
                          <small>{bytes(f.bytes)} ↗</small>
                        </a>
                      ))}
                    </div>
                  </div>
                  {detail.statistics && (
                    <LiveStatus status={detail.statistics} />
                  )}
                  <details>
                    <summary>Full metadata and configuration</summary>
                    <pre>{JSON.stringify(detail, null, 2)}</pre>
                  </details>
                </section>
              )}
            </>
          )}

          {page === "System / Logs" && (
            <section className="panel">
              <h2>Hardware jobs</h2>
              {!jobs.length ? (
                <Empty>No operations yet.</Empty>
              ) : (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Operation</th>
                        <th>Created</th>
                        <th>Status</th>
                        <th>Result</th>
                      </tr>
                    </thead>
                    <tbody>
                      {jobs.map((j) => (
                        <tr key={j.id}>
                          <td>
                            <button
                              className="text-button"
                              onClick={() => setSelectedJob(j.id)}
                            >
                              {j.kind}
                            </button>
                            <small>{j.id.slice(0, 12)}</small>
                          </td>
                          <td>{new Date(j.created_at).toLocaleString()}</td>
                          <td>
                            <Badge value={j.status} />
                          </td>
                          <td>
                            {j.error ||
                              (j.result?.passed === false
                                ? "Check failed"
                                : j.result?.passed === true
                                  ? "Check passed"
                                  : "")}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          )}

          {currentJob && (
            <section className="panel job-panel">
              <div className="panel-heading">
                <h2>
                  {currentJob.kind.replaceAll("-", " ")}{" "}
                  <Badge value={currentJob.status} />
                </h2>
                <div className="actions">
                  {["running", "queued"].includes(currentJob.status) &&
                    currentJob.kind !== "flash" && (
                      <button
                        onClick={() =>
                          run(() => api(`/jobs/${currentJob.id}/cancel`, {}))
                        }
                      >
                        Cancel job
                      </button>
                    )}
                  <button
                    className="text-button"
                    onClick={() => setSelectedJob("")}
                  >
                    Close ×
                  </button>
                </div>
              </div>
              {currentJob.error && (
                <div className="alert">{currentJob.error}</div>
              )}
              {currentJob.result && (
                <>
                  {typeof currentJob.result.passed === "boolean" && (
                    <div
                      className={`test-result ${currentJob.result.passed ? "pass" : "fail"}`}
                    >
                      {currentJob.result.passed ? "PASS" : "CHECK FAILED"}
                      {currentJob.result.rate_hz !== undefined && (
                        <span>
                          {number(currentJob.result.rate_hz)} Hz ·{" "}
                          {currentJob.result.packets} packets ·{" "}
                          {currentJob.result.sequence_gaps} sequence gaps ·{" "}
                          {currentJob.result.parse_errors} parse errors
                        </span>
                      )}
                      {currentJob.result.measured_fps !== undefined && (
                        <span>
                          {number(currentJob.result.measured_fps)} FPS ·{" "}
                          {currentJob.result.frames} frames ·{" "}
                          {currentJob.result.resolution?.join(" × ")}
                        </span>
                      )}
                    </div>
                  )}
                  <details open={currentJob.kind === "probe"}>
                    <summary>
                      Structured result{" "}
                      {currentJob.result.simulated ? "(synthetic)" : ""}
                    </summary>
                    <pre>{JSON.stringify(currentJob.result, null, 2)}</pre>
                  </details>
                </>
              )}
              <details open>
                <summary>Operation logs</summary>
                <pre className="terminal">
                  {jobLogs || "Waiting for log output…"}
                </pre>
              </details>
            </section>
          )}
          <footer>
            CSI Collection Lab{" "}
            <span>Raw data · Shared clock · Traceable sessions</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

function Preflight({ result }: { result: Json }) {
  return (
    <div className="preflight">
      <h3>
        Pre-recording checks{" "}
        <Badge value={result.passed ? "ready" : "failed"} />
      </h3>
      {result.receivers?.map((r: Json) => (
        <div key={r.logical_name}>
          <span>{r.logical_name}</span>
          <strong>
            {r.error || `${number(r.rate_hz)} Hz ${r.passed ? "✓" : "✕"}`}
          </strong>
        </div>
      ))}
      <div>
        <span>Camera</span>
        <strong>
          {result.camera?.error ||
            `${number(result.camera?.measured_fps)} FPS ${result.camera?.passed ? "✓" : "✕"}`}
        </strong>
      </div>
      <div>
        <span>Storage free</span>
        <strong>
          {bytes(result.storage?.free_bytes)}{" "}
          {result.storage?.passed ? "✓" : "✕"}
        </strong>
      </div>
    </div>
  );
}
function SessionTable({
  sessions,
  inspect,
}: {
  sessions: Json[];
  inspect: (sid: string) => void;
}) {
  return !sessions.length ? (
    <Empty>
      No recordings yet. Set up your hardware, then start your first session.
    </Empty>
  ) : (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Session</th>
            <th>Subject / Room</th>
            <th>Duration</th>
            <th>Receivers</th>
            <th>Quality</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {sessions.map((s) => (
            <tr key={s.session_id}>
              <td>
                <button
                  className="text-button"
                  onClick={() => inspect(s.session_id)}
                >
                  {s.session_id}
                </button>
                <small>
                  {s.created_at ? new Date(s.created_at).toLocaleString() : ""}
                </small>
              </td>
              <td>
                {s.configuration?.subject_id || "—"} /{" "}
                {s.configuration?.room_id || "—"}
              </td>
              <td>{age(s.duration_seconds)}</td>
              <td>{s.configuration?.receivers?.length || 0}</td>
              <td>
                <Badge value={s.quality || "unknown"} />
              </td>
              <td>
                <Badge value={s.status} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
function LiveStatus({ status }: { status: Json }) {
  return (
    <section className="panel">
      <div className="panel-heading">
        <h2>Stream health</h2>
        <span className="subtle">{bytes(status.disk_usage_bytes)} written</span>
      </div>
      {!status.receivers ? (
        <Empty>
          Receiver and camera statistics appear when recording begins.
        </Empty>
      ) : (
        <>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Receiver</th>
                  <th>Packets / saved</th>
                  <th>Current / avg Hz</th>
                  <th>RSSI</th>
                  <th>TX seq</th>
                  <th>Gaps / reset</th>
                  <th>Parse errors</th>
                  <th>Queue drops</th>
                </tr>
              </thead>
              <tbody>
                {status.receivers.map((r: Json) => (
                  <tr key={r.logical_name}>
                    <td>{r.logical_name}</td>
                    <td>
                      {r.packets} / {r.written}
                    </td>
                    <td>
                      {number(r.current_rate_hz)} / {number(r.average_rate_hz)}
                    </td>
                    <td>{r.rssi ?? "—"}</td>
                    <td>{r.latest_tx_seq ?? "—"}</td>
                    <td>
                      {r.sequence_gaps} / {r.backwards_or_resets}
                    </td>
                    <td>{r.parse_errors}</td>
                    <td>{r.queue_drops}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="metrics camera-metrics">
            <Metric
              label="Frames recorded"
              value={status.camera?.frames_recorded || 0}
            />
            <Metric
              label="Actual camera FPS"
              value={number(status.camera?.actual_fps)}
            />
            <Metric
              label="Resolution"
              value={status.camera?.resolution?.join(" × ") || "—"}
            />
            <Metric
              label="Camera queue drops"
              value={status.camera?.queue_drops || 0}
            />
          </div>
        </>
      )}
    </section>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
