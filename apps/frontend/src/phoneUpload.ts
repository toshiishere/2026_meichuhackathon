export type UploadStats = {
  frames: number;
  skipped: number;
  cameraFps: number;
  deliveredFps: number;
  encodeMs: number;
  ackMs: number;
  inFlight: number;
  maxInFlight: number;
};

// Capture/encoding is independent of acknowledgement latency. Keep both the
// encoder and network queues bounded; never build a backlog of camera frames.
export function startPhoneUpload(
  video: HTMLVideoElement,
  socket: WebSocket,
  settings: { width: number; height: number; fps: number },
  onStats: (stats: UploadStats) => void,
  onError: (error: Error) => void,
) {
  const period = 1000 / settings.fps;
  const maxInFlight = Math.min(8, Math.max(2, Math.ceil(settings.fps / 4)));
  const pending = new Map<number, number>();
  let stopped = false,
    encoding = false,
    sequence = 0,
    skipped = 0,
    delivered = 0;
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
    cancel();
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
      if (stopped || socket.readyState !== WebSocket.OPEN) return;
      if (jpeg.type !== "image/jpeg" || jpeg.size + 20 > 2 * 1024 * 1024)
        throw new Error(
          "Phone JPEG is unsupported or exceeds the 2 MiB frame limit.",
        );
      if (socket.bufferedAmount > 0) {
        skipped++;
        return;
      }
      const index = sequence++;
      const header = new ArrayBuffer(20);
      const view = new DataView(header);
      view.setUint32(0, 0x43534931); // CSI1, unchanged wire format.
      view.setUint32(4, index);
      view.setFloat64(8, captureMs);
      view.setUint32(16, skipped);
      pending.set(index, performance.now());
      // Send the encoded Blob directly; avoid copying the JPEG through an
      // ArrayBuffer and another Uint8Array before every WebSocket send.
      socket.send(new Blob([header, jpeg]));
    } catch (error) {
      fail((error as Error).message || "Phone JPEG encoding failed.");
    } finally {
      encoding = false;
    }
  }
  function capture(mediaTime: number) {
    if (stopped || socket.readyState !== WebSocket.OPEN || video.readyState < 2)
      return;
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
    if (encoding || pending.size >= maxInFlight || socket.bufferedAmount > 0) {
      skipped++;
      return;
    }
    void encodeFrame(now);
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
      fail(
        "The collector stopped acknowledging frames. Check Wi-Fi and pair again.",
      );
      return;
    }
    if (encoding && now - encodingSince > 5000) {
      fail("Phone JPEG encoding stalled. Restart the phone camera.");
      return;
    }
    onStats({
      frames: delivered,
      skipped,
      cameraFps: rate(cameraTimes, now),
      deliveredFps: rate(deliveredTimes, now),
      encodeMs,
      ackMs,
      inFlight: pending.size,
      maxInFlight,
    });
  }, 250);
  schedule();
  return {
    stop: cancel,
    ack(index: number) {
      const sentAt = pending.get(index);
      if (stopped || sentAt === undefined) return;
      const now = performance.now();
      ackMs = now - sentAt;
      pending.delete(index);
      delivered++;
      deliveredTimes.push(now);
    },
  };
}
