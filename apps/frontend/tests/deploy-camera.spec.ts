import { test, expect } from "@playwright/test";

// 1x1 JPEG: this test is about which frame is requested, not what it shows.
const jpeg = Buffer.from(
  "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKp//2Q==",
  "base64",
);

test("live camera preview follows the fused CSI clock and reports its offset", async ({
  page,
}) => {
  const capture = "c".repeat(32);
  const requested: (string | null)[] = [];
  let csiClock = 1_000_000_000_000;
  const state: any = {
    id: "run",
    status: "running",
    signal: "ready",
    capture_id: capture,
    accepted: 402,
    receiver_names: ["left", "right"],
    options: {
      model_session_id: "trained",
      source: "live",
      receivers: [{ identity: "rx1", logical_name: "left" }],
      camera: { device: "/dev/video0", width: 640, height: 480, fps: 15 },
      replay_speed: 1,
      baud_rate: 921600,
    },
    prediction: {
      label: "Walking",
      confidence: 0.7,
      scores: { Static: 0.3, Walking: 0.7 },
      source_elapsed_s: 4,
      inference_ms: 3,
      fused_receivers: ["left", "right"],
      window_end_ns: csiClock,
      receivers: [],
    },
  };
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    if (path.startsWith("/api/deploy/camera/")) {
      const timestamp = url.searchParams.get("timestamp_ns");
      requested.push(timestamp);
      await route.fulfill({
        status: 200,
        contentType: "image/jpeg",
        // The collector serves the nearest frame it holds, 40 ms after the
        // requested CSI time, and says exactly when it was captured.
        headers: {
          "X-Host-Timestamp-Ns": String(Number(timestamp) + 40_000_000),
        },
        body: jpeg,
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
    if (path === "/api/health") data = { hardware: { mode: "real" } };
    if (path === "/api/deploy/catalog")
      data = { models: [], sources: [], errors: [] };
    if (path === "/api/deploy/status") {
      csiClock += 200_000_000;
      state.source_timestamp_ns = csiClock;
      data = state;
    }
    if (path === "/api/deploy/stop") data = { ...state, status: "stopped" };
    await route.fulfill({ json: data });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Deploy", exact: true }).click();
  await expect(page.getByAltText("Deployment live camera")).toBeVisible();
  await expect(
    page.getByText("Camera frame offset from the fused CSI clock: 40 ms"),
  ).toBeVisible();
  // Each preview is fetched for the current fused CSI time, never "latest".
  await expect.poll(() => requested.length).toBeGreaterThan(2);
  expect(requested.every((value) => value && Number(value) > 0)).toBe(true);
  expect(new Set(requested).size).toBeGreaterThan(1);
  expect(Number(requested[requested.length - 1])).toBeLessThanOrEqual(csiClock);
  expect(Number(requested[requested.length - 1])).toBeGreaterThan(
    csiClock - 2_000_000_000,
  );
});
