import { test, expect } from "@playwright/test";

test("HTML proxy failures are readable and sessions reload after live reconnection", async ({ page }) => {
  let offline = true;
  await page.addInitScript(() => {
    // Control the live connection without interrupting the real collector.
    (window as any).EventSource = class {
      onmessage: any;
      onerror: any;
      constructor() { (window as any).testEvents = this; }
      close() {}
    };
  });
  await page.route("**/api/**", async (route) => {
    if (offline) {
      await route.fulfill({ status: 502, contentType: "text/html", body: "<html><h1>Bad Gateway</h1></html>" });
      return;
    }
    const path = new URL(route.request().url()).pathname;
    let data: any = [];
    if (path === "/api/health") data = { hardware: { mode: "synthetic" } };
    if (path === "/api/sessions") data = [{ session_id: "recovered_session", status: "complete" }];
    await route.fulfill({ json: data });
  });
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("Server unavailable (HTTP 502)");
  await expect(page.getByRole("alert")).not.toContainText("Unexpected token");
  await page.evaluate(() => (window as any).testEvents.onerror());
  offline = false;
  await page.evaluate(() => (window as any).testEvents.onmessage({
    data: JSON.stringify({ collection: { status: "idle" }, jobs: [] }),
  }));
  await expect(page.getByRole("alert")).toHaveCount(0);
  await expect(page.getByText("Live connection unavailable.", { exact: false })).toHaveCount(0);
  await page.getByRole("button", { name: "Sessions", exact: true }).click();
  await expect(page.getByRole("button", { name: "recovered_session", exact: true })).toBeVisible();
});
