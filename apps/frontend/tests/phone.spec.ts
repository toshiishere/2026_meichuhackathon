import { test, expect } from "@playwright/test";

test("pair a phone over HTTPS, record synchronized video, and remove its session", async ({
  page,
  browser,
  request,
}) => {
  test.skip(
    !process.env.PHONE_BASE_URL,
    "Enable the phone HTTPS override and set PHONE_BASE_URL",
  );
  test.setTimeout(90000);
  const health = await (await request.get("/api/health")).json();
  expect(health.hardware.mode).toBe("synthetic");
  const ports = await (await request.get("/api/hardware/serial")).json();
  for (const [port, logical_name, role] of [
    ["synthetic://tx", "tx_main", "csi_sender"],
    ["synthetic://rx0", "rx_left", "csi_receiver"],
  ]) {
    const device = ports.find((p: any) => p.port === port);
    const saved = await request.post("/api/devices", {
      data: {
        identity: device.identity,
        port,
        logical_name,
        role,
        target: "esp32c3",
      },
    });
    expect(saved.ok()).toBeTruthy();
  }
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(page.getByText("Services connected")).toBeVisible();
  await page
    .getByRole("button", { name: "Hardware Setup", exact: true })
    .click();
  await page
    .getByLabel("Phone web address (HTTPS)")
    .fill(process.env.PHONE_BASE_URL!);
  await page.getByLabel("Phone camera name").fill("Test phone");
  await page.getByRole("button", { name: "Create phone pairing link" }).click();
  const link = await page
    .getByLabel("Phone pairing link", { exact: true })
    .inputValue();
  const phoneContext = await browser.newContext({
    ignoreHTTPSErrors: true,
    permissions: ["camera"],
    viewport: { width: 390, height: 844 },
    isMobile: true,
  });
  const phone = await phoneContext.newPage();
  phone.on("pageerror", (e) => errors.push(e.message));
  try {
    // The externally reachable portal must not expose collection controls or files.
    for (const path of [
      "/api/sessions",
      "/api/devices",
      "/api/hardware/phone/pair",
    ]) {
      expect(
        (
          await phoneContext.request.get(`${process.env.PHONE_BASE_URL}${path}`)
        ).status(),
      ).toBe(404);
    }
    expect(
      (
        await phoneContext.request.get(
          `${process.env.PHONE_BASE_URL}/phone-ca.crt`,
        )
      ).status(),
    ).toBe(200);
    await phone.goto(link);
    expect(await phone.evaluate(() => window.isSecureContext)).toBe(true);
    await phone.getByRole("button", { name: "Start phone camera" }).click();
    await expect(phone.getByRole("status")).toContainText(
      "Streaming 640 × 480",
      { timeout: 15000 },
    );
    await expect(phone.locator(".phone-counters")).not.toContainText(
      /^0 frames delivered/,
    );
    expect(new URL(phone.url()).hash).toBe("");
    expect(
      await phone.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await page
      .getByRole("button", { name: "Refresh connected cameras" })
      .click();
    await expect(
      page
        .getByLabel("Capture device")
        .locator("option")
        .filter({ hasText: "Test phone" }),
    ).toHaveCount(1);
    const phoneDevice = await page
      .getByLabel("Capture device")
      .locator("option")
      .filter({ hasText: "Test phone" })
      .getAttribute("value");
    await page.getByLabel("Capture device").selectOption(phoneDevice!);
    await expect(page.getByLabel("width", { exact: true })).toHaveValue("640");
    await expect(page.getByLabel("width", { exact: true })).toBeDisabled();
    await page
      .getByRole("button", { name: "Data Collection", exact: true })
      .click();
    const sid = `phone_browser_${Date.now()}`;
    await page.getByLabel("Session id", { exact: true }).fill(sid);
    await page
      .getByRole("button", { name: "Run preflight", exact: true })
      .click();
    await expect(page.locator(".preflight .badge")).toHaveText("ready", {
      timeout: 20000,
    });
    await page
      .getByRole("button", { name: "Start recording", exact: false })
      .click();
    await expect(page.locator(".sticky .badge").first()).toHaveText(
      "recording",
      { timeout: 20000 },
    );
    await expect(async () => {
      const state = await (await request.get("/api/collection/status")).json();
      expect(state.camera.frames_recorded).toBeGreaterThanOrEqual(30);
    }).toPass({ timeout: 12000 });
    await page
      .getByRole("button", { name: "Stop recording", exact: false })
      .click();
    await expect(page.locator(".sticky .badge").first()).toHaveText(
      "complete",
      { timeout: 15000 },
    );
    const session = await (await request.get(`/api/sessions/${sid}`)).json();
    expect(session.configuration.camera.device).toBe(phoneDevice);
    expect(session.clock.phone_capture_clock).toContain("unaligned");
    await page.getByRole("button", { name: "Sessions", exact: true }).click();
    await page.getByRole("button", { name: sid, exact: true }).click();
    await expect(async () => {
      expect(
        await page
          .locator("video")
          .evaluate((v: HTMLVideoElement) => v.readyState),
      ).toBeGreaterThanOrEqual(2);
    }).toPass({ timeout: 10000 });
    const row = page
      .getByRole("row")
      .filter({ has: page.getByRole("button", { name: sid, exact: true }) });
    await row
      .getByRole("button", { name: `Remove session ${sid}`, exact: true })
      .click();
    const dialog = page.getByRole("dialog");
    await dialog.getByLabel("Type the session ID to confirm").fill("wrong-id");
    await expect(
      dialog.getByRole("button", { name: "Permanently remove" }),
    ).toBeDisabled();
    await dialog.getByRole("button", { name: "Cancel", exact: true }).click();
    await expect(row).toBeVisible();
    await row
      .getByRole("button", { name: `Remove session ${sid}`, exact: true })
      .click();
    await dialog.getByLabel("Type the session ID to confirm").fill(sid);
    const cameraTest = await request.post("/api/hardware/camera/test", {
      data: { device: phoneDevice, width: 640, height: 480, fps: 15 },
    });
    expect(cameraTest.ok()).toBeTruthy();
    const cameraJob = await cameraTest.json();
    await dialog.getByRole("button", { name: "Permanently remove" }).click();
    await expect(dialog.getByRole("alert")).toContainText("Hardware is busy");
    await expect(async () => {
      const jobs = await (await request.get("/api/jobs")).json();
      const job = jobs.find((j: any) => j.id === cameraJob.id);
      expect(job.status).toBe("completed");
    }).toPass({ timeout: 10000 });
    await dialog.getByRole("button", { name: "Permanently remove" }).click();
    await expect(row).toHaveCount(0, { timeout: 10000 });
    expect((await request.get(`/api/sessions/${sid}`)).status()).toBe(404);
    expect(
      (await request.get(`/api/sessions/${sid}/files/raw/video.mp4`)).status(),
    ).toBe(404);
    await phone.getByRole("button", { name: "Stop phone camera" }).click();
    await expect(phone.getByRole("status")).toContainText("Camera stopped");
    await expect(async () => {
      const cameras = await (await request.get("/api/hardware/cameras")).json();
      expect(cameras.some((c: any) => c.device === phoneDevice)).toBe(false);
    }).toPass({ timeout: 5000 });
    expect(errors).toEqual([]);
  } finally {
    await phoneContext.close();
  }
});
