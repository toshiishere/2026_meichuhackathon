import { test, expect } from "@playwright/test";

test("Train selects sessions, labels, fine-tunes, auto trains and exposes results", async ({
  page,
}) => {
  let state: any = { jobs: [], labels_ready: false, model: null };
  let active: string | null = null;
  const requests: any[] = [];
  let healthReady = true;
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    let data: any = [];
    if (path === "/api/health")
      data = { hardware: { mode: "synthetic", defaults: {} } };
    if (path === "/api/events") {
      await route.fulfill({
        contentType: "text/event-stream",
        body: 'data: {"collection":{"status":"idle"},"jobs":[]}\n\n',
      });
      return;
    }
    if (path === "/api/sessions")
      data = [
        { session_id: "train_a", status: "complete" },
        { session_id: "unfinished", status: "recording" },
      ];
    if (path === "/api/collection/status") data = { status: "idle" };
    if (path === "/api/train/health")
      data = {
        gpu: {
          ready: healthReady,
          device: "AMD Radeon test",
          rocm: "10.0.0",
          error: "ROCm unavailable",
        },
        active_job: active,
      };
    if (path === "/api/train/sessions/train_a") data = state;
    if (path === "/api/train/sessions/unfinished")
      data = { jobs: [], labels_ready: false };
    if (path.endsWith("/start")) {
      requests.push(route.request().postDataJSON());
      const job = {
        id: "a".repeat(32),
        action: requests.at(-1).action,
        status: "running",
        created_at: "2026-09-19",
      };
      state.jobs = [job];
      active = job.id;
      data = job;
    }
    if (path.endsWith("/logs"))
      data = { logs: "Labeling → preprocessing → fine-tuning" };
    if (path.endsWith("/cancel")) {
      active = null;
      state.jobs[0].status = "cancelled";
      data = { cancel_requested: true };
    }
    await route.fulfill({ json: data });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Train", exact: true }).click();
  await expect(page.getByLabel("Training session")).toHaveValue("train_a");
  await expect(
    page.getByRole("button", { name: "Fine-tune", exact: true }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "Labeling", exact: true }).click();
  await expect.poll(() => requests.at(-1)?.action).toBe("label");
  await expect(
    page.getByRole("button", { name: "Auto train", exact: true }),
  ).toBeDisabled();
  state.labels_ready = true;
  state.jobs[0].status = "completed";
  active = null;
  await expect(
    page.getByRole("button", { name: "Fine-tune", exact: true }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Fine-tune", exact: true }).click();
  await expect.poll(() => requests.at(-1)?.action).toBe("finetune");
  state.jobs[0].status = "completed";
  active = null;
  state.model = {
    run_id: "a".repeat(32),
    model_path: `train/runs/${"a".repeat(32)}/finetuned_resnet18.pth`,
    preprocessing: { windows: 20 },
    metrics: { validation_available: false },
  };
  await expect(
    page.getByRole("link", { name: "Download fine-tuned model" }),
  ).toHaveAttribute(
    "href",
    `/api/sessions/train_a/files/${state.model.model_path}`,
  );
  await expect(
    page.getByText("No independent validation intervals", { exact: false }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Auto train", exact: true }).click();
  await expect.poll(() => requests.at(-1)?.action).toBe("auto");
  await page
    .getByRole("button", { name: "Cancel training", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Labeling", exact: true }),
  ).toBeEnabled();
  await page.getByLabel("Training session").selectOption("unfinished");
  await expect(
    page.getByRole("button", { name: "Auto train", exact: true }),
  ).toBeDisabled();
  await page.getByLabel("Training session").selectOption("train_a");
  healthReady = false;
  await expect(page.getByText("ROCm unavailable")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Auto train", exact: true }),
  ).toBeDisabled();
});
