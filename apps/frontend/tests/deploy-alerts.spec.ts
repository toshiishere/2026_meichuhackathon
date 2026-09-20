import { test, expect } from "@playwright/test";

test("the operator turns fall alerts on and off while a deployment runs", async ({
  page,
}) => {
  const toggles: any[] = [];
  const state: any = {
    id: "run",
    status: "running",
    signal: "ready",
    notify: false,
    receiver_names: ["left", "right"],
    options: {
      model_session_id: "trained",
      source: "replay",
      replay_session_id: "recorded",
      replay_receivers: ["left", "right"],
      replay_speed: 4,
      camera: null,
      notify: false,
    },
    walking_seconds: 18.4,
    falls: [
      {
        fell_at_s: 44.5,
        still_seconds: 3.2,
        detail: "replay of recorded session recorded",
        notified: false,
        error: null,
      },
    ],
    prediction: {
      label: "Static",
      confidence: 0.8,
      scores: { Falling: 0.2, Static: 0.8 },
      source_elapsed_s: 47.7,
      inference_ms: 3,
      fused_receivers: ["left", "right"],
      receivers: [],
    },
  };
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
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
      data = { models: [], sources: [], errors: [] };
    if (path === "/api/deploy/notify") {
      const body = route.request().postDataJSON();
      toggles.push(body.enabled);
      state.notify = body.enabled;
      state.falls[0].notified = body.enabled;
      data = state;
    }
    if (path === "/api/deploy/status") data = state;
    await route.fulfill({ json: data });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Deploy", exact: true }).click();

  // A detected fall is always listed; sending it is the operator's choice.
  await expect(page.getByText("Falls detected (1; alerts off)")).toBeVisible();
  await expect(page.getByRole("cell", { name: "not sent" })).toBeVisible();
  const toggle = page.getByLabel(
    "Send a Discord alert when a fall is detected",
  );
  await expect(toggle).not.toBeChecked();
  await toggle.check();
  await expect.poll(() => toggles).toEqual([true]);
  await expect(page.getByRole("cell", { name: "sent" })).toBeVisible();
  await toggle.uncheck();
  await expect.poll(() => toggles).toEqual([true, false]);
  await expect(toggle).not.toBeChecked();
});

test("the walking total counts up and is reported when the run ends", async ({
  page,
}) => {
  const state: any = {
    id: "run",
    status: "running",
    signal: "ready",
    notify: true,
    receiver_names: ["left"],
    options: {
      model_session_id: "trained",
      source: "replay",
      replay_session_id: "recorded",
      replay_receivers: ["left"],
      replay_speed: 4,
      camera: null,
      notify: true,
    },
    falls: [],
    walking_seconds: 12.3,
    walking: null,
  };
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
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
      data = { models: [], sources: [], errors: [] };
    if (path === "/api/deploy/status") data = state;
    await route.fulfill({ json: data });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Deploy", exact: true }).click();

  await expect(page.getByText("Walking time so far: 12.3s")).toBeVisible();
  // Stopping the deployment settles the total and sends it to Discord.
  state.status = "completed";
  state.walking_seconds = 31.7;
  state.walking = {
    seconds: 31.7,
    detail: "replay",
    notified: true,
    error: null,
  };
  await expect(
    page.getByText(
      "Walking total for this deployment: 31.7s · Discord summary sent",
    ),
  ).toBeVisible();
});
