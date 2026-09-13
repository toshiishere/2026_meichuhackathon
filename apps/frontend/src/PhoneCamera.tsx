import React, { useEffect, useRef, useState } from "react";
import { QRCodeSVG } from "qrcode.react";
import { startPhoneUpload, type UploadStats } from "./phoneUpload";

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
  const credentials = useRef(new URLSearchParams(location.hash.slice(1)));
  const secure =
    window.isSecureContext && !!navigator.mediaDevices?.getUserMedia;

  function stop(
    message = "Camera stopped. Create a new pairing link to reconnect.",
  ) {
    generation.current++;
    upload.current?.stop();
    upload.current = null;
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
    const hidden = () => {
      if (document.hidden)
        stop(
          "Camera paused because this page was hidden. Pair again to resume.",
        );
    };
    document.addEventListener("visibilitychange", hidden);
    return () => {
      document.removeEventListener("visibilitychange", hidden);
      stop();
    };
  }, []);

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
      if (current === generation.current)
        stop("Phone camera access ended. Pair again to reconnect.");
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

  function start() {
    if (!secure) {
      setError(
        "Phone camera access needs trusted HTTPS. Open the secure phone link after installing the lab certificate.",
      );
      return;
    }
    const id = credentials.current.get("id");
    const token = credentials.current.get("token");
    if (!id || !token) {
      setError(
        "Open a fresh pairing link from Hardware Setup on the computer.",
      );
      return;
    }
    setError("");
    setBusy(true);
    setStatus("Connecting to the collector…");
    setFrames(0);
    setSkipped(0);
    setUploadStats(null);
    const current = ++generation.current;
    const ws = new WebSocket(
      `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/api/phone/stream`,
    );
    socket.current = ws;
    let settings: any;
    ws.onopen = () => ws.send(JSON.stringify({ id, token }));
    ws.onerror = () => {
      if (current === generation.current) {
        stop();
        setError(
          "Could not connect. Check the phone address, trusted HTTPS certificate, and Wi-Fi.",
        );
      }
    };
    ws.onclose = () => {
      if (current === generation.current)
        stop("Disconnected. Create a new phone pairing link on the computer.");
    };
    ws.onmessage = async (event) => {
      if (current !== generation.current) return;
      const data = JSON.parse(event.data);
      try {
        if (data.type === "error") {
          stop();
          setError(data.message);
          return;
        }
        if (data.type === "settings") {
          settings = data;
          setStatus("Allow camera access on this phone…");
          if (!(await openCamera(cameraChoice, current, data))) return;
          ws.send(
            JSON.stringify({
              type: "ready",
              native_resolution: [
                video.current!.videoWidth,
                video.current!.videoHeight,
              ],
            }),
          );
        }
        if (data.type === "ready") {
          setBusy(false);
          setActive(true);
          setStatus(
            `Streaming ${settings.width} × ${settings.height} at up to ${settings.fps} FPS`,
          );
          // The fragment is never sent in HTTP requests or server access logs.
          history.replaceState(null, "", location.pathname);
          (navigator as any).wakeLock
            ?.request("screen")
            .then((lock: any) => {
              if (current === generation.current) wakeLock.current = lock;
              else lock.release();
            })
            .catch(() => {});
          upload.current = startPhoneUpload(
            video.current!,
            ws,
            settings,
            (stats) => {
              if (current !== generation.current) return;
              setFrames(stats.frames);
              setSkipped(stats.skipped);
              setUploadStats(stats);
            },
            (error) => {
              if (current !== generation.current) return;
              stop();
              setError(error.message);
            },
          );
        }
        if (data.type === "ack") {
          upload.current?.ack(data.frame_idx);
        }
      } catch (e) {
        if (current === generation.current) {
          stop();
          setError((e as Error).message);
        }
      }
    };
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
            onClick={start}
          >
            Start phone camera
          </button>
          <button
            disabled={!active && !busy && !previewReady}
            className="danger"
            onClick={() =>
              stop(
                active || socket.current
                  ? undefined
                  : "Camera preview stopped.",
              )
            }
          >
            Stop phone camera
          </button>
        </div>
        <div className="phone-counters">
          <span>{frames} frames delivered</span>
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
          select this camera, run preflight, and start recording. Host
          timestamps measure frame arrival; Wi-Fi and JPEG encoding add delay.
        </p>
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
      const data = await response.json();
      if (!response.ok)
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail),
        );
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
