import { test, expect } from "@playwright/test";

test("phone pairing settings win over defaults on initial selection, refresh and camera switches", async ({
  page,
}) => {
  const first = {
    device: `phone://${"a".repeat(32)}`,
    name: "Paired phone",
    width: 640,
    height: 480,
    fps: 15,
  };
  const second = {
    device: `phone://${"b".repeat(32)}`,
    name: "Second phone",
    width: 1280,
    height: 720,
    fps: 30,
  };
  let cameras = [first, second];
  const jobs: any[] = [];
  await page.route("**/api/events", (route) =>
    route.fulfill({
      contentType: "text/event-stream",
      body: `data: ${JSON.stringify({ collection: { status: "idle" }, jobs })}\n\n`,
    }),
  );
  await page.route("**/api/jobs/*/logs", (route) =>
    route.fulfill({ json: { logs: "" } }),
  );
  await page.route("**/api/hardware/cameras", (route) =>
    route.fulfill({ json: cameras }),
  );
  const testRequests: any[] = [];
  await page.route("**/api/hardware/camera/test", async (route) => {
    testRequests.push(route.request().postDataJSON());
    const job = {
      id: "f".repeat(32),
      kind: "camera-test",
      status: "completed",
      result: { passed: true },
    };
    jobs.splice(0, jobs.length, job);
    await route.fulfill({ json: job });
  });
  await page.goto("/");
  await page
    .getByRole("button", { name: "Hardware Setup", exact: true })
    .click();
  const device = page.getByLabel("Capture device");
  async function expectSettings(width: number, height: number, fps: number) {
    await expect(page.getByLabel("width", { exact: true })).toHaveValue(
      String(width),
    );
    await expect(page.getByLabel("height", { exact: true })).toHaveValue(
      String(height),
    );
    await expect(page.getByLabel("Requested FPS", { exact: true })).toHaveValue(
      String(fps),
    );
    await expect(page.getByLabel("width", { exact: true })).toBeDisabled();
  }
  // No selectOption call: initial discovery must apply the phone preset.
  await expect(device).toHaveValue(first.device);
  await expectSettings(640, 480, 15);
  await page.getByRole("button", { name: "Test camera", exact: true }).click();
  await expect.poll(() => testRequests.length).toBe(1);
  expect(testRequests[0]).toMatchObject({
    device: first.device,
    width: 640,
    height: 480,
    fps: 15,
  });
  await page.getByRole("button", { name: "Close ×", exact: true }).click();

  await device.selectOption(second.device);
  await expectSettings(1280, 720, 30);
  // Refreshed discovery is authoritative even without changing the selection.
  cameras = [first, { ...second, fps: 15 }];
  await page.getByRole("button", { name: "↻ Refresh", exact: true }).click();
  await expectSettings(1280, 720, 15);
  await page.getByRole("button", { name: "Test camera", exact: true }).click();
  await expect.poll(() => testRequests.length).toBe(2);
  expect(testRequests[1]).toMatchObject({
    device: second.device,
    width: 1280,
    height: 720,
    fps: 15,
  });
  await page.getByRole("button", { name: "Close ×", exact: true }).click();
  await page
    .getByRole("button", { name: "Data Collection", exact: true })
    .click();
  await expect(
    page.getByText("Camera: 1280 × 720 at 15 FPS.", { exact: false }),
  ).toBeVisible();

  await page
    .getByRole("button", { name: "Hardware Setup", exact: true })
    .click();
  cameras = [first];
  await page.getByRole("button", { name: "↻ Refresh", exact: true }).click();
  await expect(device.locator("option:checked")).toContainText("Disconnected");
  await device.selectOption(first.device);
  await expectSettings(640, 480, 15);
});
