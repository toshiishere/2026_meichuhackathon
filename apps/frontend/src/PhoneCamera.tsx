import React, { useEffect, useRef, useState } from "react";

export function PhoneCamera() {
  const [status, setStatus] = useState("Ready to connect");
  const [error, setError] = useState("");
  const [active, setActive] = useState(false);
  const [busy, setBusy] = useState(false);
  const [frames, setFrames] = useState(0);
  const [skipped, setSkipped] = useState(0);
  const [facing, setFacing] = useState("environment");
  const video = useRef<HTMLVideoElement>(null);
  const socket = useRef<WebSocket | null>(null);
  const media = useRef<MediaStream | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const wakeLock = useRef<any>(null);
  const generation = useRef(0);
  const credentials = useRef(new URLSearchParams(location.hash.slice(1)));
  const secure =
    window.isSecureContext && !!navigator.mediaDevices?.getUserMedia;

  function stop(
    message = "Camera stopped. Create a new pairing link to reconnect.",
  ) {
    generation.current++;
    if (timer.current) clearInterval(timer.current);
    timer.current = null;
    socket.current?.close();
    socket.current = null;
    media.current?.getTracks().forEach((track) => track.stop());
    media.current = null;
    if (video.current) video.current.srcObject = null;
    wakeLock.current?.release().catch(() => {});
    wakeLock.current = null;
    setActive(false);
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
    const current = ++generation.current;
    const ws = new WebSocket(
      `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/api/phone/stream`,
    );
    socket.current = ws;
    let settings: any,
      sending = false,
      sequence = 0,
      skippedCount = 0,
      awaitingSince = 0;
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d")!;
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
          const stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
              facingMode: { ideal: facing },
              width: { ideal: data.width },
              height: { ideal: data.height },
              frameRate: { ideal: data.fps, max: data.fps },
            },
          });
          if (current !== generation.current) {
            stream.getTracks().forEach((t) => t.stop());
            return;
          }
          media.current = stream;
          stream.getVideoTracks()[0].onended = () => {
            if (current === generation.current)
              stop("Phone camera access ended. Pair again to reconnect.");
          };
          video.current!.srcObject = stream;
          await video.current!.play();
          if (current !== generation.current) return;
          canvas.width = data.width;
          canvas.height = data.height;
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
          timer.current = setInterval(() => {
            if (
              current !== generation.current ||
              ws.readyState !== WebSocket.OPEN
            )
              return;
            if (sending || ws.bufferedAmount > 0) {
              skippedCount++;
              setSkipped(skippedCount);
              if (performance.now() - awaitingSince > 5000) {
                stop();
                setError(
                  "The collector stopped acknowledging frames. Check Wi-Fi and pair again.",
                );
              }
              return;
            }
            const v = video.current!;
            if (v.readyState < 2) return;
            sending = true;
            awaitingSince = performance.now();
            const captureMs = performance.now();
            context.fillStyle = "black";
            context.fillRect(0, 0, canvas.width, canvas.height);
            const scale = Math.min(
              canvas.width / v.videoWidth,
              canvas.height / v.videoHeight,
            );
            const w = v.videoWidth * scale,
              h = v.videoHeight * scale;
            context.drawImage(
              v,
              (canvas.width - w) / 2,
              (canvas.height - h) / 2,
              w,
              h,
            );
            canvas.toBlob(
              async (blob) => {
                if (
                  current !== generation.current ||
                  ws.readyState !== WebSocket.OPEN
                )
                  return;
                if (!blob) {
                  stop();
                  setError("Phone JPEG encoding failed.");
                  return;
                }
                const jpeg = await blob.arrayBuffer();
                if (
                  current !== generation.current ||
                  ws.readyState !== WebSocket.OPEN
                )
                  return;
                const bytes = new Uint8Array(20 + jpeg.byteLength);
                bytes.set([67, 83, 73, 49]);
                const header = new DataView(bytes.buffer);
                header.setUint32(4, sequence++);
                header.setFloat64(8, captureMs);
                header.setUint32(16, skippedCount);
                bytes.set(new Uint8Array(jpeg), 20);
                ws.send(bytes);
              },
              "image/jpeg",
              0.8,
            );
          }, 1000 / settings.fps);
        }
        if (data.type === "ack") {
          sending = false;
          setFrames(data.frame_idx + 1);
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
          <span>Camera direction</span>
          <select
            aria-label="Camera direction"
            disabled={active || busy}
            value={facing}
            onChange={(e) => setFacing(e.target.value)}
          >
            <option value="environment">Rear camera</option>
            <option value="user">Front camera</option>
          </select>
        </label>
        <div className="actions">
          <button
            className="primary"
            disabled={active || busy || !secure}
            onClick={start}
          >
            Start phone camera
          </button>
          <button
            disabled={!active && !busy}
            className="danger"
            onClick={() => stop()}
          >
            Stop phone camera
          </button>
        </div>
        <div className="phone-counters">
          <span>{frames} frames delivered</span>
          <span>{skipped} upload intervals skipped</span>
        </div>
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
