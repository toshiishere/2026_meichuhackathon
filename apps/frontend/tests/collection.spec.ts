import { test, expect } from "@playwright/test";

test("configure synthetic boards, verify streams, record and inspect artifacts", async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(page.getByText("Services connected")).toBeVisible();
  await page
    .getByRole("button", { name: "Hardware Setup", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "ESP32 boards" }),
  ).toBeVisible();
  for (const [port, name, role] of [
    ["synthetic://tx", "tx_main", "csi_sender"],
    ["synthetic://rx0", "rx_left", "csi_receiver"],
    ["synthetic://rx1", "rx_right", "csi_receiver"],
  ]) {
    await page.getByLabel("Selected serial port").selectOption(port);
    await page.getByLabel("Logical name", { exact: true }).fill(name);
    await page.getByLabel("Role", { exact: true }).selectOption(role);
    await page.getByRole("button", { name: "Save assignment" }).click();
    await expect(page.getByRole("cell", { name, exact: true })).toBeVisible();
  }
  await page
    .getByRole("button", { name: "CSI test / monitor", exact: true })
    .click();
  await expect(page.locator(".test-result")).toContainText("PASS", {
    timeout: 20000,
  });
  await page.getByRole("button", { name: "Close ×", exact: true }).click();
  await page.getByLabel("width", { exact: true }).fill("320");
  await page.getByLabel("height", { exact: true }).fill("240");
  await page.getByRole("button", { name: "Live preview", exact: true }).click();
  await expect(page.getByAltText("Camera preview")).toBeVisible();
  await page
    .getByRole("button", { name: "Close preview", exact: true })
    .click();
  // The streaming request releases its hardware lease when the disconnect reaches the service.
  await expect(async () => {
    const response = await request.post("/api/hardware/camera/test", {
      data: { device: "synthetic://camera", width: 320, height: 240, fps: 30 },
    });
    expect(response.status()).toBe(200);
  }).toPass({ timeout: 10000 });
  await expect(async () => {
    const jobs = await (await request.get("/api/jobs")).json();
    expect(
      jobs.find((j: any) => j.kind === "camera-test")?.result?.passed,
    ).toBe(true);
  }).toPass({ timeout: 15000 });
  await page
    .getByRole("button", { name: "Data Collection", exact: true })
    .click();
  const sid = `browser_${Date.now()}`;
  await page.getByLabel("Session id", { exact: true }).fill(sid);
  await page.getByLabel("Subject id", { exact: true }).fill("P01");
  await page.getByLabel("Room id", { exact: true }).fill("roomA");
  await page
    .getByRole("button", { name: "Run preflight", exact: true })
    .click();
  await expect(page.locator(".preflight .badge")).toHaveText("ready", {
    timeout: 20000,
  });
  await page
    .getByRole("button", { name: "Start recording", exact: false })
    .click();
  await expect(page.locator(".sticky .badge").first()).toHaveText("recording", {
    timeout: 20000,
  });
  await expect(async () => {
    expect(
      Number(
        await page
          .locator(".camera-metrics .metric")
          .first()
          .locator("strong")
          .innerText(),
      ),
    ).toBeGreaterThanOrEqual(60);
  }).toPass({ timeout: 10000 });
  await page
    .getByRole("button", { name: "Stop recording", exact: false })
    .click();
  await expect(page.locator(".sticky .badge").first()).toHaveText("complete", {
    timeout: 15000,
  });
  await page.getByRole("button", { name: "Sessions", exact: true }).click();
  await page.getByRole("button", { name: sid, exact: true }).click();
  await expect(page.locator("video")).toBeVisible();
  await expect(async () => {
    expect(
      await page
        .locator("video")
        .evaluate((v: HTMLVideoElement) => v.readyState),
    ).toBeGreaterThanOrEqual(2);
  }).toPass({ timeout: 10000 });
  await page.locator("video").evaluate((v: HTMLVideoElement) => {
    v.muted = true;
    return v.play();
  });
  await expect(async () => {
    expect(
      await page
        .locator("video")
        .evaluate((v: HTMLVideoElement) => v.currentTime),
    ).toBeGreaterThan(0);
  }).toPass({ timeout: 5000 });
  await expect(
    page.getByRole("link", { name: "raw/video_frames.parquet", exact: false }),
  ).toBeVisible();
  const data = await (await request.get(`/api/sessions/${sid}`)).json();
  expect(data.status).toBe("complete");
  expect(data.configuration.subject_id).toBe("P01");
  expect(
    data.artifacts.some((x: any) => x.path === "raw/csi_rx_left.csv.zst"),
  ).toBe(true);
  const video = await request.get(`/api/sessions/${sid}/files/raw/video.mp4`, {
    headers: { Range: "bytes=0-99" },
  });
  expect(video.status()).toBe(206);
  expect(errors).toEqual([]);
  await page.screenshot({
    path: "test-results/csi-sessions.png",
    fullPage: true,
  });
});
