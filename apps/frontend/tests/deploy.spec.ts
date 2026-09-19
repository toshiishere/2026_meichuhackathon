import { test, expect } from "@playwright/test";

test("Deploy fuses the chosen receivers, shows predictions, stops and starts live capture", async ({
  page,
}) => {
  const starts: any[] = [];
  const model = {
    session_id: "trained",
    run_id: "a".repeat(32),
    classes: ["Static", "Walking"],
    preprocessing: { window_seconds: 2, sample_rate_hz: 100 },
  };
  let state: any = { status: "idle" };
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    let data: any = [];
    if (path === "/api/events") {
      await route.fulfill({
        contentType: "text/event-stream",
        body: 'data: {"collection":{"status":"idle"},"jobs":[]}\n\n',
      });
      return;
    }
    if (path === "/api/health") data = { hardware: { mode: "real" } };
    if (path === "/api/devices")
      data = [
        {
          identity: "rx1",
          logical_name: "left",
          role: "csi_receiver",
          port: "/dev/ttyACM0",
        },
        {
          identity: "rx2",
          logical_name: "right",
          role: "csi_receiver",
          port: "/dev/ttyACM2",
        },
      ];
    if (path === "/api/hardware/serial")
      data = [
        { identity: "rx1", port: "/dev/ttyACM1" },
        { identity: "rx2", port: "/dev/ttyACM2" },
      ];
    if (path === "/api/hardware/cameras")
      data = [
        {
          device: "phone://" + "b".repeat(32),
          name: "Phone",
          width: 640,
          height: 480,
          fps: 15,
        },
      ];
    if (path === "/api/deploy/catalog")
      data = {
        models: [model],
        sources: [
          { session_id: "trained", receivers: ["left"] },
          { session_id: "other", receivers: ["left", "mid", "right"] },
        ],
        errors: [],
      };
    if (path === "/api/deploy/status") data = state;
    if (path === "/api/deploy/start") {
      const options = route.request().postDataJSON();
      starts.push(options);
      state = {
        status: "running",
        signal: "ready",
        options,
        accepted: 603,
        receiver_names: ["left", "right"],
        prediction: {
          label: "Walking",
          confidence: 0.6,
          scores: { Static: 0.4, Walking: 0.6 },
          source_elapsed_s: 2,
          inference_ms: 4,
          fused_receivers: ["left", "right"],
          uncovered_receivers: [],
          receivers: [
            {
              receiver: "left",
              label: "Static",
              confidence: 0.9,
              scores: { Static: 0.9, Walking: 0.1 },
            },
            {
              receiver: "right",
              label: "Walking",
              confidence: 0.9,
              scores: { Static: 0.1, Walking: 0.9 },
            },
          ],
        },
      };
      data = state;
    }
    if (path === "/api/deploy/stop") {
      state = { ...state, status: "stopped", prediction: null };
      data = state;
    }
    await route.fulfill({ json: data });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Deploy", exact: true }).click();
  await expect(page.getByLabel("Model session")).toHaveValue("trained");
  await page.getByLabel("Replay session").selectOption("other");
  // Every recorded receiver is fused by default; one can be left out.
  await expect(page.getByLabel("Replay receiver left")).toBeChecked();
  await expect(page.getByLabel("Replay receiver mid")).toBeChecked();
  await page.getByLabel("Replay receiver mid").uncheck();
  await page.getByLabel("Replay speed").selectOption("2");
  await page
    .getByRole("button", { name: "Start deployment", exact: true })
    .click();
  await expect(page.locator(".deploy-prediction strong")).toHaveText("Walking");
  await expect(page.getByText("Fused 2 of 2 receivers")).toBeVisible();
  await page.getByText("Per-receiver scores before fusion").click();
  await expect(page.getByRole("cell", { name: "left" })).toBeVisible();
  expect(starts[0]).toMatchObject({
    model_session_id: "trained",
    source: "replay",
    replay_session_id: "other",
    replay_receivers: ["left", "right"],
    replay_speed: 2,
    camera: null,
  });
  await expect(page.getByLabel("Model session")).toBeDisabled();
  await page
    .getByRole("button", { name: "Stop deployment", exact: true })
    .click();
  await page.getByLabel("CSI source").selectOption("live");
  await expect(page.getByLabel("Live receiver left")).toBeChecked();
  await expect(page.getByLabel("Live receiver right")).toBeChecked();
  await page
    .getByLabel("Live camera (optional)")
    .selectOption("phone://" + "b".repeat(32));
  await page
    .getByRole("button", { name: "Start deployment", exact: true })
    .click();
  await expect.poll(() => starts.length).toBe(2);
  expect(starts[1]).toMatchObject({
    source: "live",
    receivers: [
      { logical_name: "left", identity: "rx1", port: "/dev/ttyACM1" },
      { logical_name: "right", identity: "rx2", port: "/dev/ttyACM2" },
    ],
    camera: { width: 640, height: 480, fps: 15 },
    replay_session_id: null,
  });
  await page.getByRole("button", { name: "Dashboard", exact: true }).click();
  await page.getByRole("button", { name: "Deploy", exact: true }).click();
  await expect(page.getByLabel("CSI source")).toHaveValue("live");
  await expect(
    page.getByRole("button", { name: "Stop deployment", exact: true }),
  ).toBeEnabled();
  await page
    .getByRole("button", { name: "Stop deployment", exact: true })
    .click();
});
