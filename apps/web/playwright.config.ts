import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  outputDir: "./test-results",
  webServer: {
    command: "pnpm dev --port 3000",
    url: "http://127.0.0.1:3000",
    reuseExistingServer: true,
    timeout: 120_000
  },
  use: {
    baseURL: "http://127.0.0.1:3000",
    trace: "on-first-retry"
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 1000 } } },
    { name: "mobile", use: { ...devices["Pixel 7"] } }
  ]
});
