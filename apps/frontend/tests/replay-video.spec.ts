import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";

// A real H.264 fixture exercises browser decoding, autoplay, rate and seeking.
const movie = readFileSync(new URL("./fixtures/replay.mp4", import.meta.url));

test("replay video follows the server playhead, speed, stop and reconnect", async ({
  page,
}) => {
  let state: any = { status: "idle" };
  let began = 0;
  const requested: string[] = [];
  // The worker's playhead, as the page can never observe it directly: the
  // browser only ever sees samples that are already a round trip old.
  const playhead = () => Math.min(9, 3 + ((Date.now() - began) / 1000) * 2);
  const model = {
    session_id: "demo",
    classes: ["Static", "Walking"],
    preprocessing: { window_seconds: 2, sample_rate_hz: 100 },
  };
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith("/video.mp4")) {
      requested.push(path);
      const range = /bytes=(\d+)-(\d*)/.exec(
        route.request().headers().range || "",
      );
      const start = range ? Number(range[1]) : 0;
      const end = range?.[2] ? Number(range[2]) : movie.length - 1;
      await route.fulfill({
        status: range ? 206 : 200,
        contentType: "video/mp4",
        headers: {
          "Accept-Ranges": "bytes",
          ...(range
            ? { "Content-Range": `bytes ${start}-${end}/${movie.length}` }
            : {}),
        },
        body: movie.subarray(start, end + 1),
      });
      return;
    }
    if (path === "/api/events") {
      await route.fulfill({
        contentType: "text/event-stream",
        body: 'data: {"collection":{"status":"idle"},"jobs":[]}\n\n',
      });
      return;
    }
    let data: any = [];
    if (path === "/api/health") data = { hardware: { mode: "synthetic" } };
    if (path === "/api/deploy/catalog")
      data = {
        models: [model],
        sources: [{ session_id: "demo", receivers: ["left"] }],
        errors: [],
      };
    if (path === "/api/deploy/start") {
      began = Date.now();
      state = {
        id: "run",
        status: "running",
        options: route.request().postDataJSON(),
        video_available: true,
        video_path: "derived/video.mp4",
        video_status: "ready",
        video_time_s: 3,
        video_playing: true,
      };
      data = state;
    }
    if (path === "/api/deploy/status") {
      if (state.status === "running") {
        state.video_time_s = playhead();
        state.video_playing = state.video_time_s < 9;
      }
      data = state;
    }
    if (path === "/api/deploy/stop") {
      state.status = "stopped";
      data = state;
    }
    await route.fulfill({ json: data });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Deploy", exact: true }).click();
  await page.getByLabel("Replay speed").selectOption("2");
  await page
    .getByRole("button", { name: "Start deployment", exact: true })
    .click();
  const video = page.getByLabel("Synchronized replay video");
  await expect(video).toBeVisible();
  await expect
    .poll(() => video.evaluate((v: HTMLVideoElement) => v.readyState))
    .toBeGreaterThanOrEqual(2);
  await expect
    .poll(() => video.evaluate((v: HTMLVideoElement) => v.paused))
    .toBe(false);
  // Playback runs at the replay speed, trimmed slightly to absorb drift.
  const rate = await video.evaluate((v: HTMLVideoElement) => v.playbackRate);
  expect(rate).toBeGreaterThan(1.7);
  expect(rate).toBeLessThan(2.3);
  // Track the live playhead, not the last polled sample: the page extrapolates
  // between polls, so it must stay close to where the worker actually is.
  await expect
    .poll(async () =>
      Math.abs(
        (await video.evaluate((v: HTMLVideoElement) => v.currentTime)) -
          playhead(),
      ),
    )
    .toBeLessThan(0.35);
  await page.getByRole("button", { name: "Dashboard", exact: true }).click();
  await page.getByRole("button", { name: "Deploy", exact: true }).click();
  await expect
    .poll(async () =>
      Math.abs(
        (await video.evaluate((v: HTMLVideoElement) => v.currentTime)) -
          playhead(),
      ),
    )
    .toBeLessThan(0.35);
  await page
    .getByRole("button", { name: "Stop deployment", exact: true })
    .click();
  await expect
    .poll(() => video.evaluate((v: HTMLVideoElement) => v.paused))
    .toBe(true);
  await expect
    .poll(async () =>
      Math.abs(
        (await video.evaluate((v: HTMLVideoElement) => v.currentTime)) -
          state.video_time_s,
      ),
    )
    .toBeLessThan(0.1);
  // The worker decides which file is seekable; the page plays that one.
  expect(requested[0]).toBe(
    "/api/sessions/demo/files/derived/video.mp4",
  );
});
