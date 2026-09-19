import { test, expect } from "@playwright/test";

test("incomplete session recovery requires confirmation and shows recovered provenance", async ({
  page,
}) => {
  let recovered = false;
  let requests = 0;
  const session = () => ({
    session_id: "partial",
    status: recovered ? "complete" : "incomplete",
    recovered,
    quality: recovered ? "recovered" : "incomplete",
    artifacts: [],
  });
  const job = {
    id: "a".repeat(32),
    kind: "session-recover",
    status: "completed",
    result: { session_id: "partial", recovered: true },
  };
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/events") {
      await route.fulfill({
        contentType: "text/event-stream",
        body: `retry: 200\ndata: ${JSON.stringify({ collection: { status: "idle" }, jobs: recovered ? [job] : [] })}\n\n`,
      });
      return;
    }
    let data: any = [];
    if (path === "/api/health") data = { hardware: { mode: "synthetic" } };
    if (path === "/api/sessions") data = [session()];
    if (path === "/api/sessions/partial") data = session();
    if (path.endsWith("/logs")) data = { logs: "Recovered temporary files" };
    if (path === "/api/sessions/partial/recover") {
      requests++;
      recovered = true;
      data = job;
    }
    await route.fulfill({ json: data });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Sessions", exact: true }).click();
  page.once("dialog", (dialog) => dialog.dismiss());
  await page
    .getByRole("button", { name: "Recover session partial", exact: true })
    .click();
  expect(requests).toBe(0);
  page.once("dialog", (dialog) => dialog.accept());
  await page
    .getByRole("button", { name: "Recover session partial", exact: true })
    .click();
  await expect.poll(() => requests).toBe(1);
  await expect(
    page.getByText("Recovered from temporary files.", { exact: false }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Recover session partial", exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("heading", { name: "partial", exact: true }),
  ).toBeVisible();
});
