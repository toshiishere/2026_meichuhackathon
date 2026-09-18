import { test, expect } from "@playwright/test";

test.use({
  launchOptions: { args: ["--use-fake-device-for-media-stream=fps=30"] },
});

for (const [width, height, fps, fallback] of [
  [640, 480, 15, false],
  [1280, 720, 30, false],
  [1280, 720, 30, true],
] as const) {
  test(`phone ${width}x${height} ${fps} FPS survives 120ms acknowledgements (fallback=${fallback})`, async ({
    browser,
    request,
  }) => {
    test.skip(
      !process.env.PHONE_BASE_URL,
      "Requires isolated phone HTTPS services",
    );
    expect(
      (await (await request.get("/api/health")).json()).hardware.mode,
    ).toBe("synthetic");
    const response = await request.post("/api/hardware/phone/pair", {
      data: { name: "Latency regression", width, height, fps },
    });
    expect(response.ok()).toBe(true);
    const pair = await response.json();
    const context = await browser.newContext({
      ignoreHTTPSErrors: true,
      permissions: ["camera"],
    });
    const page = await context.newPage();
    const errors: string[] = [];
    const timers = new Set<ReturnType<typeof setTimeout>>();
    let dropAcks = false,
      outstanding = 0,
      peak = 0,
      connections = 0;
    page.on("pageerror", (error) => errors.push(error.message));
    if (fallback)
      await page.addInitScript(() => {
        Object.defineProperty(window, "OffscreenCanvas", { value: undefined });
      });
    await page.routeWebSocket("**/api/phone/stream", (ws) => {
      connections++;
      const server = ws.connectToServer();
      ws.onMessage((message) => {
        if (typeof message !== "string") {
          outstanding++;
          peak = Math.max(peak, outstanding);
        }
        server.send(message);
      });
      server.onMessage((message) => {
        if (typeof message === "string" && JSON.parse(message).type === "ack") {
          if (dropAcks) return;
          const timer = setTimeout(() => {
            timers.delete(timer);
            outstanding--;
            ws.send(message);
          }, 120);
          timers.add(timer);
        } else ws.send(message);
      });
    });
    try {
      await page.goto(
        `${process.env.PHONE_BASE_URL}/phone#${new URLSearchParams({ id: pair.id, token: pair.token })}`,
      );
      await page
        .getByRole("button", { name: "Start phone camera", exact: true })
        .click();
      await expect(page.getByRole("status")).toContainText("Streaming");
      const jobResponse = await request.post("/api/hardware/camera/test", {
        data: { device: `phone://${pair.id}`, width, height, fps },
      });
      expect(jobResponse.ok()).toBe(true);
      const job = await jobResponse.json();
      let result: any;
      await expect(async () => {
        const jobs = await (await request.get("/api/jobs")).json();
        result = jobs.find((j: any) => j.id === job.id);
        expect(result.status).toBe("completed");
      }).toPass({ timeout: 15000 });
      console.log(
        JSON.stringify({
          width,
          height,
          fps,
          fallback,
          measured_fps: result.result.measured_fps,
          peak_in_flight: peak,
        }),
      );
      expect(result.result.measured_fps).toBeGreaterThan(fps * 0.85);
      expect(result.result.resolution).toEqual([width, height]);
      expect(peak).toBeGreaterThan(1);
      expect(peak).toBeLessThanOrEqual(Math.ceil(fps / 4));
      // Lost acknowledgements must leave frames in persistent storage and
      // reconnect automatically; they must not stop camera capture.
      dropAcks = true;
      await expect
        .poll(() => connections, { timeout: 8000 })
        .toBeGreaterThan(1);
      dropAcks = false;
      await expect(page.getByRole("status")).toContainText("Streaming", {
        timeout: 10000,
      });
      await page
        .getByRole("button", { name: "Stop phone camera", exact: true })
        .click();
      await expect(page.getByRole("status")).toContainText(
        "All frames uploaded",
        { timeout: 15000 },
      );
      expect(errors).toEqual([]);
    } finally {
      for (const timer of timers) clearTimeout(timer);
      await context.close();
    }
  });
}
