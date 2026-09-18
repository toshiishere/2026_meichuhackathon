import { test, expect, type WebSocketRoute } from "@playwright/test";

test.use({
  launchOptions: { args: ["--use-fake-device-for-media-stream=fps=30"] },
});

test("phone buffers an outage, survives reload, and finishes a session stopped offline", async ({
  browser,
  request,
}) => {
  test.skip(
    !process.env.PHONE_BASE_URL,
    "Requires isolated phone HTTPS services",
  );
  test.setTimeout(90000);
  expect((await (await request.get("/api/health")).json()).hardware.mode).toBe(
    "synthetic",
  );
  const pair = await (
    await request.post("/api/hardware/phone/pair", {
      data: { name: "Resilient phone", width: 640, height: 480, fps: 15 },
    })
  ).json();
  const context = await browser.newContext({
    ignoreHTTPSErrors: true,
    permissions: ["camera"],
  });
  const page = await context.newPage();
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  let offline = false;
  const connections = new Set<WebSocketRoute>();
  await page.routeWebSocket("**/api/phone/stream", (ws) => {
    if (offline) {
      ws.close({ code: 1012 });
      return;
    }
    const server = ws.connectToServer();
    connections.add(ws);
    connections.add(server);
  });
  const sid = `phone_recovery_${Date.now()}`;
  try {
    await page.goto(
      `${process.env.PHONE_BASE_URL}/phone#${new URLSearchParams({ id: pair.id, token: pair.token })}`,
    );
    await page
      .getByRole("button", { name: "Start phone camera", exact: true })
      .click();
    await expect(page.getByRole("status")).toContainText("Streaming");
    const started = await request.post("/api/collection/start", {
      data: {
        session_id: sid,
        receivers: [{ logical_name: "rx_left", port: "synthetic://rx0" }],
        camera: {
          device: `phone://${pair.id}`,
          width: 640,
          height: 480,
          fps: 15,
        },
        expected_rate_hz: 100,
      },
    });
    expect(started.ok(), await started.text()).toBe(true);
    await expect(async () => {
      const state = await (await request.get("/api/collection/status")).json();
      expect(state.status).toBe("recording");
      expect(state.camera.frames_recorded).toBeGreaterThan(20);
    }).toPass({ timeout: 20000 });
    offline = true;
    for (const connection of connections) connection.close({ code: 1012 });
    await expect(page.getByRole("status")).toContainText("Disconnected");
    const queued = () =>
      page.evaluate(async () => {
        return new Promise<number>((resolve, reject) => {
          const open = indexedDB.open("csi-phone-buffer", 1);
          open.onsuccess = () => {
            const db = open.result;
            const query = db
              .transaction("streams")
              .objectStore("streams")
              .getAll();
            query.onsuccess = () => {
              resolve(query.result[0]?.count || 0);
              db.close();
            };
            query.onerror = () => reject(query.error);
          };
        });
      });
    // This exceeds the former five-second stall timeout while CSI keeps recording.
    await expect(async () => expect(await queued()).toBeGreaterThan(90)).toPass(
      { timeout: 15000 },
    );
    expect(
      (await (await request.get("/api/collection/status")).json()).status,
    ).toBe("recording");
    expect(
      (await request.post("/api/collection/stop", { data: {} })).ok(),
    ).toBe(true);
    await expect(async () => {
      const state = await (await request.get("/api/collection/status")).json();
      expect(state.status).toBe("stopping");
      expect(state.camera.phone_sync_wait_seconds).toBeGreaterThan(0);
    }).toPass({ timeout: 5000 });
    const savedBeforeReload = await queued();
    await page.reload();
    await expect(
      page.getByRole("button", { name: /Resume Resilient phone/ }),
    ).toBeVisible();
    expect(await queued()).toBeGreaterThanOrEqual(savedBeforeReload);
    offline = false;
    await page.getByRole("button", { name: /Resume Resilient phone/ }).click();
    await expect(page.getByRole("status")).toContainText("Streaming");
    await expect(async () => {
      const state = await (await request.get("/api/collection/status")).json();
      expect(state.status).toBe("complete");
      expect(state.camera.phone_reconnects).toBeGreaterThanOrEqual(1);
      expect(state.camera.queue_drops).toBe(0);
      expect(state.camera.phone_queue_drops).toBe(0);
    }).toPass({ timeout: 20000 });
    const session = await (await request.get(`/api/sessions/${sid}`)).json();
    expect(session.statistics.camera.frames_recorded).toBeGreaterThan(110);
    expect(session.errors).toEqual([]);
    expect(session.clock.phone_clock_offset_ns).toBeGreaterThan(0);
    await page
      .getByRole("button", { name: "Stop phone camera", exact: true })
      .click();
    await expect(page.getByRole("status")).toHaveText(
      "Camera stopped. All frames uploaded.",
    );
    expect(await queued()).toBe(0);
    expect(errors).toEqual([]);
  } finally {
    await context.close();
  }
});
