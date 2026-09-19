import React, { useEffect, useRef, useState } from "react";
import {
  PhoneBuffer,
  pendingPhoneStreams,
  type PhoneBufferMeta,
} from "./phoneBuffer";
import { QRCodeSVG } from "qrcode.react";
import { startPhoneUpload, type UploadStats } from "./phoneUpload";
import { readApiResponse } from "./api";

function wideCameraScore(camera: MediaDeviceInfo) {
  const label = camera.label;
  if (/front|user|selfie|前置|前置鏡頭|前置镜头/i.test(label)) return 0;
  if (/ultra[\s-]*wide|0[.,]5\s*[x×]|超廣角|超广角/i.test(label)) return 2;
  return /\bwide\b|廣角|广角/i.test(label) ? 1 : 0;
}

export function PhoneCamera() {
  const [status, setStatus] = useState("Ready to connect");
  const [error, setError] = useState("");
  const [active, setActive] = useState(false);
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState<PhoneBufferMeta[]>([]);
  const [finishing, setFinishing] = useState(false);
  const [frames, setFrames] = useState(0);
  const [skipped, setSkipped] = useState(0);
  const [uploadStats, setUploadStats] = useState<UploadStats | null>(null);
  const [cameraChoice, setCameraChoice] = useState("auto");
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [cameraLabel, setCameraLabel] = useState("");
  const [previewReady, setPreviewReady] = useState(false);
  const video = useRef<HTMLVideoElement>(null);
  const socket = useRef<WebSocket | null>(null);
  const media = useRef<MediaStream | null>(null);
  const upload = useRef<ReturnType<typeof startPhoneUpload> | null>(null);
  const wakeLock = useRef<any>(null);
  const generation = useRef(0);
  const buffer = useRef<PhoneBuffer | null>(null);
  const retry = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const heartbeat = useRef<ReturnType<typeof setInterval> | undefined>(
    undefined,
  );
  const finishRequested = useRef(false);
  const reconnect = useRef<(() => void) | null>(null);
  const releaseLock = useRef<(() => void) | null>(null);
  const refreshSaved = () =>
    pendingPhoneStreams()
      .then(setSaved)
      .catch((e) => setError(`Phone storage unavailable: ${e.message}`));
  const keepAwake = () => {
    const current = generation.current;
    (navigator as any).wakeLock
      ?.request("screen")
      .then((lock: any) => {
        if (current !== generation.current) {
          void lock.release();
          return;
        }
        wakeLock.current?.release().catch(() => {});
        wakeLock.current = lock;
      })
      .catch(() => {});
  };
  const credentials = useRef(new URLSearchParams(location.hash.slice(1)));
  const secure =
    window.isSecureContext && !!navigator.mediaDevices?.getUserMedia;

  function stop(
    message = "Camera stopped. Saved frames remain available below.",
  ) {
    generation.current++;
    clearTimeout(retry.current);
    clearInterval(heartbeat.current);
    reconnect.current = null;
    finishRequested.current = false;
    setFinishing(false);
    upload.current?.stop();
    upload.current = null;
    const oldBuffer = buffer.current;
    buffer.current = null;
    const unlock = releaseLock.current;
    releaseLock.current = null;
    void (oldBuffer?.close() || Promise.resolve()).finally(() => {
      unlock?.();
      void refreshSaved();
    });
    socket.current?.close();
    socket.current = null;
    media.current?.getTracks().forEach((track) => track.stop());
    media.current = null;
    if (video.current) video.current.srcObject = null;
    wakeLock.current?.release().catch(() => {});
    wakeLock.current = null;
    setActive(false);
    setPreviewReady(false);
    setCameraLabel("");
    setBusy(false);
    setStatus(message);
  }
  useEffect(() => {
    void refreshSaved();
    const visible = () => {
      if (!document.hidden && upload.current) keepAwake();
    };
    document.addEventListener("visibilitychange", visible);
    return () => {
      document.removeEventListener("visibilitychange", visible);
      stop();
    };
  }, []);

  async function finish() {
    if (!upload.current) {
      stop("Camera preview stopped.");
      return;
    }
    finishRequested.current = true;
    setFinishing(true);
    setStatus(
      "Finishing capture and uploading saved frames… Keep this page open.",
    );
    await upload.current.pauseCapture();
    media.current?.getTracks().forEach((track) => track.stop());
    media.current = null;
  }

  async function openCamera(
    choice: string,
    current: number,
    settings = { width: 1280, height: 720, fps: 30 },
  ) {
    // Release the previous lens first: phones often cannot open two at once.
    media.current?.getTracks().forEach((track) => track.stop());
    media.current = null;
    setPreviewReady(false);
    const dimensions = {
      width: { ideal: settings.width },
      height: { ideal: settings.height },
      frameRate: { ideal: settings.fps, max: settings.fps },
    };
    const acquire = async (deviceId?: string) => {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          ...dimensions,
          ...(deviceId
            ? { deviceId: { exact: deviceId } }
            : {
                facingMode: {
                  ideal: choice === "user" ? "user" : "environment",
                },
              }),
        },
      });
      if (current !== generation.current) {
        stream.getTracks().forEach((track) => track.stop());
        return null;
      }
      media.current = stream;
      return stream;
    };
    let stream = await acquire(
      choice.startsWith("device:") ? choice.slice(7) : undefined,
    );
    if (!stream) return null;
    // Camera labels and additional lenses usually become available only after
    // permission is granted. No pairing token is consumed by a local preview.
    const available = (await navigator.mediaDevices.enumerateDevices()).filter(
      (device) => device.kind === "videoinput" && device.deviceId,
    );
    if (current !== generation.current) return null;
    setCameras(available);
    if (choice === "auto") {
      const preferred = [...available]
        .filter((device) => wideCameraScore(device) > 0)
        .sort((a, b) => wideCameraScore(b) - wideCameraScore(a))[0];
      if (
        preferred &&
        preferred.deviceId !== stream.getVideoTracks()[0].getSettings().deviceId
      ) {
        stream.getTracks().forEach((track) => track.stop());
        try {
          stream = await acquire(preferred.deviceId);
        } catch {
          if (current !== generation.current) return null;
          // Keep automatic mode usable if an advertised lens cannot be opened.
          stream = await acquire();
        }
        if (!stream) return null;
      }
    }
    const track = stream.getVideoTracks()[0];
    track.onended = () => {
      if (current === generation.current) {
        void upload.current?.pauseCapture();
        setError(
          "Camera access ended. Saved frames will continue uploading. Uncaptured video cannot be recovered.",
        );
      }
    };
    video.current!.srcObject = stream;
    await video.current!.play();
    if (current !== generation.current) return null;
    setCameraLabel(track.label || "Selected camera");
    setPreviewReady(true);
    return stream;
  }

  async function previewCamera(choice = cameraChoice) {
    if (!secure || active || busy) return;
    setError("");
    setBusy(true);
    setStatus("Allow camera access to choose a lens…");
    const current = ++generation.current;
    try {
      if (await openCamera(choice, current))
        setStatus(
          "Preview only. Choose a camera, then start to connect to the collector.",
        );
    } catch (e) {
      if (current === generation.current) {
        stop("Camera unavailable. Choose another camera and try again.");
        setError((e as Error).message);
      }
    } finally {
      if (current === generation.current) setBusy(false);
    }
  }

  async function start(resumeId?: string) {
    if (!secure) {
      setError("Phone camera access needs trusted HTTPS.");
      return;
    }
    const id = resumeId || credentials.current.get("id");
    const token = credentials.current.get("token");
    if (!id) {
      setError("Open a pairing link or resume a saved camera below.");
      return;
    }
    setError("");
    setBusy(true);
    setStatus("Connecting to the collector…");
    setFrames(0);
    setSkipped(0);
    setUploadStats(null);
    finishRequested.current = false;
    setFinishing(false);
    const current = ++generation.current;
    const valid = () => current === generation.current;
    try {
      // One tab owns a buffer. Holding a Web Lock also protects sequence numbers
      // across a reload/recovery tab without a stale timeout-based lease.
      if (navigator.locks) {
        await new Promise<void>((resolve, reject) => {
          void navigator.locks
            .request(`csi-phone:${id}`, { ifAvailable: true }, async (lock) => {
              if (!lock) {
                reject(
                  new Error("This camera is already open in another tab."),
                );
                return;
              }
              await new Promise<void>((release) => {
                releaseLock.current = release;
                resolve();
              });
            })
            .catch(reject);
        });
      }
      const previous = (await pendingPhoneStreams()).find(
        (item) => item.id === id,
      );
      if (!valid()) {
        releaseLock.current?.();
        return;
      }
      if (previous) buffer.current = await PhoneBuffer.open(id);
      if (!previous && !token)
        throw new Error("Open a fresh pairing link from the computer.");
      void navigator.storage?.persist?.().catch(() => {});
      let attempts = 0;
      function connect() {
        if (!valid()) return;
        const ws = new WebSocket(
          `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/api/phone/stream`,
        );
        socket.current = ws;
        let ready = false,
          lastReceived = performance.now(),
          lastSequence = -1;
        let samples: { probe: string; mid: number; rtt: number }[] = [];
        let messages = Promise.resolve();
        const send = (data: unknown) => {
          if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(data));
        };
        const ours = () => valid() && socket.current === ws;
        ws.onopen = () =>
          send({
            id,
            protocol: 2,
            ...(buffer.current?.meta.resumeToken
              ? { resume_token: buffer.current.meta.resumeToken }
              : { token }),
          });
        const retryConnection = () => {
          if (!ours()) return;
          clearInterval(heartbeat.current);
          socket.current = null;
          ws.close();
          upload.current?.detach();
          setStatus(
            finishRequested.current
              ? "Disconnected. Saved frames will upload when the collector reconnects…"
              : "Disconnected. Saving frames on this phone and reconnecting…",
          );
          retry.current = setTimeout(
            connect,
            Math.min(5000, 500 * 2 ** Math.min(attempts++, 4)),
          );
        };
        reconnect.current = retryConnection;
        ws.onerror = retryConnection;
        ws.onclose = retryConnection;
        const activate = async () => {
          if (!ours()) return;
          const store = buffer.current!;
          if (!upload.current) {
            upload.current = startPhoneUpload(
              video.current!,
              store,
              store.meta.settings,
              (stats) => {
                if (!valid()) return;
                setFrames(stats.frames);
                setSkipped(stats.skipped);
                setUploadStats(stats);
              },
              (e) => {
                if (valid())
                  setError(
                    `Capture paused: ${e.message} Saved frames will continue uploading.`,
                  );
              },
              () => reconnect.current?.(),
            );
          }
          await upload.current.attach(ws, lastSequence);
          if (!ours()) {
            upload.current?.detach();
            return;
          }
          ready = true;
          attempts = 0;
          setBusy(false);
          setActive(true);
          finishRequested.current ||= store.meta.paused;
          setFinishing(finishRequested.current);
          const settings = store.meta.settings;
          setStatus(
            finishRequested.current
              ? "Uploading saved frames… Keep this page open."
              : `Streaming ${settings.width} × ${settings.height} at up to ${settings.fps} FPS`,
          );
          history.replaceState(null, "", location.pathname);
          keepAwake();
        };
        // Heartbeats detect dead sockets even while no frames can get through.
        heartbeat.current = setInterval(() => {
          if (!ours()) return;
          if (performance.now() - lastReceived > (ready ? 10000 : 120000)) {
            retryConnection();
            return;
          }
          if (ready && upload.current) {
            const progress = upload.current.progress();
            send({
              type:
                finishRequested.current &&
                buffer.current!.meta.paused &&
                !progress.buffered_frames
                  ? "finish"
                  : "heartbeat",
              ...progress,
            });
          }
        }, 2000);
        ws.onmessage = (event) => {
          lastReceived = performance.now();
          messages = messages
            .then(async () => {
              if (!ours()) return;
              const data = JSON.parse(event.data);
              if (data.type === "error") throw new Error(data.message);
              if (data.type === "settings") {
                if (!buffer.current)
                  buffer.current = await PhoneBuffer.open(id!, data);
                await buffer.current.update({ resumeToken: data.resume_token });
                if (!ours()) return;
                if (!media.current && !buffer.current.meta.paused) {
                  setStatus("Allow camera access on this phone…");
                  if (!(await openCamera(cameraChoice, current, data))) return;
                }
                send({
                  type: "ready",
                  native_resolution: media.current
                    ? [video.current!.videoWidth, video.current!.videoHeight]
                    : [data.width, data.height],
                });
              } else if (data.type === "ready") {
                await buffer.current!.update({
                  resumeToken: data.resume_token,
                });
                lastSequence = data.last_sequence;
                if (data.ended) {
                  await buffer.current!.acknowledge(lastSequence);
                  await buffer.current!.complete();
                  stop("Camera stopped. All frames uploaded.");
                } else if (data.clock_ready) await activate();
                else send({ type: "clock", client_ms: buffer.current!.now() });
              } else if (data.type === "clock") {
                const received = buffer.current!.now();
                samples.push({
                  probe: data.probe,
                  mid: (data.client_ms + received) / 2,
                  rtt: received - data.client_ms,
                });
                if (samples.length < 3)
                  send({ type: "clock", client_ms: buffer.current!.now() });
                else {
                  const best = samples.sort((a, b) => a.rtt - b.rtt)[0];
                  send({
                    type: "clock_commit",
                    probe: best.probe,
                    client_mid_ms: best.mid,
                  });
                }
              } else if (data.type === "synced") await activate();
              else if (data.type === "ack")
                await upload.current?.ack(data.frame_idx);
              else if (data.type === "finished") {
                await buffer.current!.complete();
                stop("Camera stopped. All frames uploaded.");
              }
            })
            .catch((e) => {
              if (ours()) {
                stop();
                setError(e.message);
              }
            });
        };
      }
      connect();
    } catch (e) {
      if (valid()) {
        stop();
        setError((e as Error).message);
      }
    }
  }

  return (
    <main className="phone-page">
      <section className="panel">
        <div className="eyebrow">CSI COLLECTION LAB</div>
        <h1>Phone camera</h1>
        <p>
          Use this phone as the camera for your recording. Audio is never
          captured.
        </p>
        {!secure && (
          <div className="notice warning">
            Camera permission requires trusted HTTPS. Use the secure address
            from Hardware Setup.
          </div>
        )}
        {error && (
          <div role="alert" className="alert">
            {error}
          </div>
        )}
        <video
          ref={video}
          autoPlay
          playsInline
          muted
          aria-label="Phone camera view"
        />
        <p role="status">{status}</p>
        <label className="field">
          <span>Camera</span>
          <select
            aria-label="Phone camera selection"
            disabled={active || busy}
            value={cameraChoice}
            onChange={(e) => {
              const choice = e.target.value;
              setCameraChoice(choice);
              if (previewReady) void previewCamera(choice);
            }}
          >
            <option value="auto">Automatic · prefer rear wide-angle</option>
            <option value="environment">Default rear camera</option>
            <option value="user">Front camera</option>
            {cameras.map((camera, index) => (
              <option key={camera.deviceId} value={`device:${camera.deviceId}`}>
                {camera.label || `Camera ${index + 1}`}
                {wideCameraScore(camera) > 0 ? " · wide-angle" : ""}
              </option>
            ))}
          </select>
        </label>
        {cameraLabel && <p className="hint">Using: {cameraLabel}</p>}
        <p className="hint">
          Tap Choose camera / preview to allow access and list the available
          lenses. Automatic prefers an identified ultra-wide camera, then
          wide-angle, then the default rear camera. Some browsers expose only a
          subset of phone cameras. Choose your lens before starting the stream.
        </p>
        <div className="actions">
          <button
            disabled={active || busy || !secure}
            onClick={() => void previewCamera()}
          >
            Choose camera / preview
          </button>
          <button
            className="primary"
            disabled={active || busy || !secure}
            onClick={() => void start()}
          >
            Start phone camera
          </button>
          <button
            disabled={finishing || (!active && !busy && !previewReady)}
            className="danger"
            onClick={() => void finish().catch((e) => setError(e.message))}
          >
            Stop phone camera
          </button>
        </div>
        <div className="phone-counters">
          <span>{frames} frames delivered</span>
          <span aria-label="Buffered phone frames">
            {uploadStats?.bufferedFrames || 0} frames saved locally ·{" "}
            {((uploadStats?.bufferedBytes || 0) / 1048576).toFixed(1)} MiB
            awaiting upload
          </span>
          <span>{skipped} frames skipped before upload</span>
        </div>
        {uploadStats && (
          <div
            className="phone-counters"
            aria-label="Phone streaming performance"
          >
            <span>Camera: {uploadStats.cameraFps.toFixed(1)} FPS</span>
            <span>Delivered: {uploadStats.deliveredFps.toFixed(1)} FPS</span>
            <span>JPEG: {uploadStats.encodeMs.toFixed(0)} ms</span>
            <span>Acknowledgement: {uploadStats.ackMs.toFixed(0)} ms</span>
            <span>
              In flight: {uploadStats.inFlight}/{uploadStats.maxInFlight}
            </span>
          </div>
        )}
        <p className="hint">
          Keep this page visible and the phone awake. Return to the computer to
          select this camera, run preflight, and start recording. Connection
          losses are retried automatically; up to 512 MiB of unacknowledged
          frames are saved on this phone (subject to available browser storage).
          Keep this page open until uploads finish. Closing or locking the
          browser can stop capture. Capture timestamps use an estimated
          phone-to-collector clock offset; original capture and arrival times
          are also retained.
        </p>
        {!active && !busy && saved.length > 0 && (
          <div className="actions">
            {saved.map((item) => (
              <button key={item.id} onClick={() => void start(item.id)}>
                Resume {item.settings.name} · {item.count} saved frames
              </button>
            ))}
          </div>
        )}
        <a href="/phone-ca.crt">Download lab CA certificate</a>
      </section>
    </main>
  );
}

export function PhoneSetup({
  onConnected,
  phoneOrigin,
}: {
  onConnected: () => void;
  phoneOrigin?: string;
}) {
  const [origin, setOrigin] = useState(
    location.protocol === "https:"
      ? location.origin
      : `https://${location.hostname}:8443`,
  );
  useEffect(() => {
    if (phoneOrigin) setOrigin(phoneOrigin);
  }, [phoneOrigin]);
  const [name, setName] = useState("Phone camera");
  const [preset, setPreset] = useState("640,480,15");
  const [link, setLink] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  async function pair() {
    setBusy(true);
    setError("");
    try {
      const address = new URL(origin);
      if (
        address.protocol !== "https:" &&
        !(
          address.protocol === "http:" &&
          ["localhost", "127.0.0.1"].includes(address.hostname)
        )
      )
        throw new Error("Use an HTTPS phone address.");
      const [width, height, fps] = preset.split(",").map(Number);
      const response = await fetch("/api/hardware/phone/pair", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, width, height, fps }),
      });
      const data = await readApiResponse(response);
      setLink(
        `${address.origin}/phone#${new URLSearchParams({ id: data.id, token: data.token })}`,
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="panel">
      <div className="panel-heading">
        <h2>Connect a phone camera</h2>
        <span className="subtle">Browser camera · HTTPS</span>
      </div>
      <p>
        Open a pairing link on your phone, allow camera access, then refresh
        cameras here. The phone becomes a selectable recording camera.
      </p>
      <div className="form-grid">
        <label className="field">
          <span>Phone web address (HTTPS)</span>
          <input
            aria-label="Phone web address (HTTPS)"
            value={origin}
            onChange={(e) => setOrigin(e.target.value)}
            placeholder="https://192.168.1.50:8443"
          />
        </label>
        <label className="field">
          <span>Phone camera name</span>
          <input
            aria-label="Phone camera name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </label>
        <label className="field">
          <span>Phone capture settings</span>
          <select
            aria-label="Phone capture settings"
            value={preset}
            onChange={(e) => setPreset(e.target.value)}
          >
            <option value="640,480,15">640 × 480 · 15 FPS</option>
            <option value="1280,720,15">1280 × 720 · 15 FPS</option>
            <option value="1280,720,30">1280 × 720 · 30 FPS</option>
          </select>
        </label>
      </div>
      {error && (
        <div role="alert" className="alert">
          {error}
        </div>
      )}
      <div className="actions">
        <button
          disabled={busy || !name.trim()}
          className="primary"
          onClick={pair}
        >
          Create phone pairing link
        </button>
        <button onClick={onConnected}>Refresh connected cameras</button>
      </div>
      {link && (
        <div className="pair-link">
          <figure className="pair-qr">
            <QRCodeSVG
              value={link}
              size={256}
              level="M"
              marginSize={4}
              title="Scan to open the phone camera pairing link"
            />
            <figcaption>
              Scan with your phone to open the pairing link
            </figcaption>
          </figure>
          <label className="field">
            <span>One-use phone link · expires in 10 minutes</span>
            <input aria-label="Phone pairing link" readOnly value={link} />
          </label>
          <button
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(link);
              } catch {
                setError("Select and copy the pairing link manually.");
              }
            }}
          >
            Copy link
          </button>{" "}
          <a href={link} target="_blank" rel="noreferrer">
            Open phone page ↗
          </a>
        </div>
      )}
      <details>
        <summary>First-time HTTPS setup</summary>
        <p className="hint">
          Set <code>PHONE_HOST</code> in <code>.env</code> to this computer’s
          hostname or reachable IP address (without a scheme or port). Run{" "}
          <code>make phone-cert</code>, then <code>make phone</code>. On your
          phone download the lab certificate at{" "}
          <code>http://YOUR_PHONE_HOST:8081/phone-ca.crt</code>, install it as a
          trusted certificate, and open the HTTPS pairing link. On iOS also
          enable full trust in Certificate Trust Settings. A browser warning
          bypass alone may not enable camera access.
        </p>
      </details>
    </section>
  );
}
