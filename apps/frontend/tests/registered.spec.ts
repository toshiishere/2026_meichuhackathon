import { test, expect } from "@playwright/test";

test("registered names follow identity across port changes and can be forgotten offline", async ({
  page,
  request,
}) => {
  const ports = await (await request.get("/api/hardware/serial")).json();
  const device = ports.find((p: any) => p.port === "synthetic://tx");
  expect(device).toBeTruthy();
  const saved = await request.post("/api/devices", {
    data: {
      identity: device.identity,
      port: device.port,
      logical_name: "battery_tx",
      role: "csi_sender",
    },
  });
  expect(saved.ok()).toBe(true);
  let connected = true;
  await page.route("**/api/hardware/serial", (route) =>
    route.fulfill({
      json: ports
        .filter((p: any) => connected || p.identity !== device.identity)
        .map((p: any) =>
          p.identity === device.identity
            ? {
                ...p,
                device: "/dev/ttyACM7",
                port: "/dev/ttyACM7",
                stable_path: null,
              }
            : p,
        ),
    }),
  );
  await page.goto("/");
  await page
    .getByRole("button", { name: "Hardware Setup", exact: true })
    .click();
  const registry = page.getByRole("region", {
    name: "Registered logical names",
  });
  const row = registry
    .getByRole("row")
    .filter({
      has: page.getByRole("cell", { name: "battery_tx", exact: true }),
    });
  await expect(
    row.getByRole("cell", { name: "Connected", exact: true }),
  ).toBeVisible();
  await expect(row).toContainText("/dev/ttyACM7");
  await expect(row).toContainText(device.identity);
  connected = false;
  await page.getByRole("button", { name: "↻ Refresh", exact: true }).click();
  await expect(
    row.getByRole("cell", { name: "Disconnected", exact: true }),
  ).toBeVisible();
  await expect(row).toContainText("No current port");
  await expect(row).toContainText(`Last registered: ${device.port}`);
  await row
    .getByRole("button", { name: "Remove registration battery_tx" })
    .click();
  await expect(row).toHaveCount(0);
  expect(
    (await (await request.get("/api/devices")).json()).some(
      (b: any) => b.identity === device.identity,
    ),
  ).toBe(false);
  connected = true;
  await page.getByRole("button", { name: "↻ Refresh", exact: true }).click();
  await expect(
    page
      .getByRole("row")
      .filter({
        has: page.getByRole("button", { name: "/dev/ttyACM7", exact: true }),
      }),
  ).toContainText("Unassigned");
  await page
    .getByRole("button", { name: "Data Collection", exact: true })
    .click();
  await expect(page.getByLabel("CSI sender (optional)")).toHaveValue("");
});
