import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "tests",
  timeout: 60000,
  use: {
    baseURL: process.env.BASE_URL || "http://127.0.0.1:8080",
    headless: true,
    channel: "chrome",
    launchOptions: { args: ["--use-fake-device-for-media-stream"] },
  },
  workers: 1,
  reporter: "list",
});
