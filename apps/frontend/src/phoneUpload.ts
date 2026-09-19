import { PhoneBuffer } from "./phoneBuffer";
export type UploadStats = {
  frames: number;
  skipped: number;
  cameraFps: number;
  deliveredFps: number;
  encodeMs: number;
  ackMs: number;
  inFlight: number;
  maxInFlight: number;
  bufferedFrames: number;
  bufferedBytes: number;
};

// Persist before sending. Network retries never hold up camera capture.
export function startPhoneUpload(
  video: HTMLVideoElement,
  buffer: PhoneBuffer,
  settings: { width: number; height: number; fps: number },
  onStats: (stats: UploadStats) => void,
  onError: (error: Error) => void,
  onNetworkStall: () => void,
) {
  const period = 1000 / settings.fps;
  const maxInFlight = Math.min(8, Math.max(2, Math.ceil(settings.fps / 4)));
  const pending = new Map<number, number>();
  let stopped = false,
    encoding = false,
    paused = buffer.meta.paused,
    skipped = buffer.meta.skipped;
  let socket: WebSocket | null = null;
  let lastSent = -1,
    pumping = false,
    captureStamp = 0;
  let encodingTask: Promise<void> = Promise.resolve();
  let encodeMs = 0,
    ackMs = 0,
    lastMediaTime = -1;
  let nextDue: number | undefined;
  let encodingSince = 0,
    callbackId = 0;
  const cameraTimes: number[] = [],
    deliveredTimes: number[] = [];
  let canvas: OffscreenCanvas | HTMLCanvasElement;
  let context: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;
  const offscreen =
    typeof OffscreenCanvas !== "undefined" &&
    typeof OffscreenCanvas.prototype.convertToBlob === "function"
      ? new OffscreenCanvas(settings.width, settings.height)
      : null;
  const offscreenContext = offscreen?.getContext("2d", { alpha: false });
  if (offscreen && offscreenContext) {
    canvas = offscreen;
    context = offscreenContext;
  } else {
    canvas = document.createElement("canvas");
    canvas.width = settings.width;
    canvas.height = settings.height;
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) throw new Error("Phone camera canvas is unavailable.");
    context = ctx;
  }
  const useVideoCallback =
    typeof video.requestVideoFrameCallback === "function";
  function cancel() {
    stopped = true;
    clearInterval(watchdog);
    if (useVideoCallback) video.cancelVideoFrameCallback(callbackId);
    else cancelAnimationFrame(callbackId);
    pending.clear();
  }
  function fail(message: string) {
    if (stopped) return;
    paused = true;
    void buffer.update({ paused: true }).catch(() => {});
    onError(new Error(message));
  }
  function rate(times: number[], now: number) {
    while (times.length && times[0] < now - 2000) times.shift();
    const elapsed = times.length > 1 ? times[times.length - 1] - times[0] : 0;
    return elapsed > 0 ? ((times.length - 1) * 1000) / elapsed : 0;
  }
  async function encodeFrame(captureMs: number) {
    encoding = true;
    encodingSince = performance.now();
    captureStamp = captureMs;
    try {
      context.fillStyle = "black";
      context.fillRect(0, 0, canvas.width, canvas.height);
      const scale = Math.min(
        canvas.width / video.videoWidth,
        canvas.height / video.videoHeight,
      );
      const width = video.videoWidth * scale,
        height = video.videoHeight * scale;
      context.drawImage(
        video,
        (canvas.width - width) / 2,
        (canvas.height - height) / 2,
        width,
        height,
      );
      let jpeg: Blob;
      if (canvas instanceof HTMLCanvasElement) {
        // Older-browser fallback avoids HTMLCanvasElement.toBlob's idle-task
        // scheduling. Only one encode runs, at the requested capture cadence.
        const dataUrl = canvas.toDataURL("image/jpeg", 0.8);
        const binary = atob(dataUrl.slice(dataUrl.indexOf(",") + 1));
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        jpeg = new Blob([bytes], { type: "image/jpeg" });
      } else {
        jpeg = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.8 });
      }
      encodeMs = performance.now() - encodingSince;
      if (stopped) return;
      if (jpeg.type !== "image/jpeg" || jpeg.size + 20 > 2 * 1024 * 1024)
        throw new Error(
          "Phone JPEG is unsupported or exceeds the 2 MiB frame limit.",
        );
      await buffer.append(jpeg, captureMs, skipped);
      void pump();
    } catch (error) {
      fail((error as Error).message || "Phone JPEG encoding failed.");
    } finally {
      encoding = false;
    }
  }
  function capture(mediaTime: number) {
    if (stopped || paused || video.readyState < 2) return;
    if (mediaTime <= lastMediaTime) return;
    lastMediaTime = mediaTime;
    const now = performance.now();
    cameraTimes.push(now);
    // Use the camera timeline for pacing, with tolerance for timestamp rounding.
    // No duplicate frames or catch-up bursts are manufactured to reach a target.
    const cameraMs = mediaTime * 1000;
    if (nextDue !== undefined && cameraMs + 1 < nextDue) return;
    nextDue =
      nextDue === undefined
        ? cameraMs + period
        : nextDue +
          Math.max(1, Math.floor((cameraMs - nextDue) / period) + 1) * period;
    if (encoding) {
      skipped++;
      return;
    }
    encodingTask = encodeFrame(buffer.now());
  }
  function schedule() {
    if (stopped) return;
    if (useVideoCallback) {
      callbackId = video.requestVideoFrameCallback((_now, metadata) => {
        capture(metadata.mediaTime);
        schedule();
      });
    } else {
      callbackId = requestAnimationFrame(() => {
        capture(video.currentTime);
        schedule();
      });
    }
  }
  const watchdog = setInterval(() => {
    const now = performance.now();
    const oldest = pending.values().next().value;
    if (oldest !== undefined && now - oldest > 5000) {
      socket = null;
      pending.clear();
      onNetworkStall();
    }
    if (encoding && now - encodingSince > 5000) {
      fail("Phone JPEG encoding stalled. Restart the phone camera.");
      return;
    }
    void pump();
    onStats({
      frames: buffer.meta.nextSequence - buffer.meta.count,
      skipped,
      cameraFps: rate(cameraTimes, now),
      deliveredFps: rate(deliveredTimes, now),
      encodeMs,
      ackMs,
      inFlight: pending.size,
      maxInFlight,
      bufferedFrames: buffer.meta.count,
      bufferedBytes: buffer.meta.bytes,
    });
  }, 250);
  async function pump() {
    if (pumping || stopped || !socket || socket.readyState !== WebSocket.OPEN)
      return;
    pumping = true;
    const ws = socket;
    try {
      const free = maxInFlight - pending.size;
      if (free <= 0 || ws.bufferedAmount > 2 * 1024 * 1024) return;
      const frames = await buffer.batch(lastSent, free);
      for (const frame of frames) {
        if (stopped || socket !== ws || ws.readyState !== WebSocket.OPEN) break;
        ws.send(frame.payload);
        lastSent = frame.sequence;
        pending.set(frame.sequence, performance.now());
      }
    } catch (e) {
      fail((e as Error).message);
    } finally {
      pumping = false;
    }
  }
  schedule();
  return {
    stop: cancel,
    detach() {
      socket = null;
      pending.clear();
    },
    async attach(ws: WebSocket, lastSequence: number) {
      if (
        !Number.isInteger(lastSequence) ||
        lastSequence < -1 ||
        lastSequence >= buffer.meta.nextSequence
      )
        throw new Error("Collector and saved phone sequence numbers disagree.");
      await buffer.acknowledge(lastSequence);
      pending.clear();
      lastSent = lastSequence;
      socket = ws;
      void pump();
    },
    async pauseCapture() {
      paused = true;
      await encodingTask;
      await buffer.update({ paused: true });
    },
    progress() {
      return {
        // Do not claim that capture has flushed past an encoding still in progress.
        capture_ms: encoding ? Math.max(0, captureStamp - 0.001) : buffer.now(),
        last_sequence: buffer.meta.nextSequence - 1,
        buffered_frames: buffer.meta.count,
      };
    },
    async ack(index: number) {
      if (stopped) return;
      if (!Number.isInteger(index) || index >= buffer.meta.nextSequence)
        throw new Error("Collector acknowledged an unknown phone frame.");
      const now = performance.now(),
        sentAt = pending.get(index);
      if (sentAt !== undefined) ackMs = now - sentAt;
      await buffer.acknowledge(index);
      for (const key of pending.keys()) if (key <= index) pending.delete(key);
      deliveredTimes.push(now);
      void pump();
    },
  };
}
