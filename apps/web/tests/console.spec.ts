import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
});

test("console renders the working product surface", async ({ page }, testInfo) => {
  await page.goto("/?fixture=1");
  await page.waitForFunction(() => window.__USB_AGENTS_READY === true);
  await expect(
    page.getByRole("heading", { name: "Compare agent runtime behavior before it ships." })
  ).toBeVisible();
  await expect(page.getByRole("button", { name: /Run embedded/ })).toBeVisible();
  await expect(page.getByRole("button", { name: /Copy run command/ })).toBeVisible();
  await expect(page.getByRole("table", { name: "Compatibility matrix" })).toBeVisible();
  await expect(page.getByLabel("Selected case inspector")).toBeVisible();
  await expect
    .poll(() => page.evaluate(() => window.__USB_AGENTS_DEBUG__?.().resultCount ?? 0))
    .toBeGreaterThan(0);

  const mainBounds = await page.locator(".main-panel").boundingBox();
  const inspectorBounds = await page.locator(".inspector").boundingBox();
  const compact = testInfo.project.name === "mobile";
  expect(mainBounds?.width ?? 0).toBeGreaterThan(compact ? 320 : 520);
  expect(inspectorBounds?.width ?? 0).toBeGreaterThan(compact ? 320 : 300);

  const horizontalOverlap =
    mainBounds && inspectorBounds
      ? Math.max(
          0,
          Math.min(mainBounds.x + mainBounds.width, inspectorBounds.x + inspectorBounds.width) -
            Math.max(mainBounds.x, inspectorBounds.x)
        )
      : 0;
  const verticalOverlap =
    mainBounds && inspectorBounds
      ? Math.max(
          0,
          Math.min(mainBounds.y + mainBounds.height, inspectorBounds.y + inspectorBounds.height) -
            Math.max(mainBounds.y, inspectorBounds.y)
        )
      : 0;
  expect(horizontalOverlap * verticalOverlap).toBe(0);
  await expect(page).toHaveScreenshot("console-surface.png", {
    fullPage: true,
    animations: "disabled"
  });
});

test("trace and artifact views are reachable", async ({ page }) => {
  await page.goto("/?fixture=1");
  await page.waitForFunction(() => window.__USB_AGENTS_READY === true);
  await page.getByRole("button", { name: "Trace" }).click();
  await expect(page.getByRole("heading", { name: "Trace rail" })).toBeVisible();
  await page.getByRole("button", { name: "Compare" }).click();
  await expect(page.getByRole("heading", { name: "Regression compare" })).toBeVisible();
  await page.getByRole("button", { name: "Artifacts" }).click();
  await expect(page.getByRole("heading", { name: "Artifact inspector" })).toBeVisible();
  await page.getByRole("button", { name: "Docs" }).click();
  await expect(page.getByRole("heading", { name: "Project cockpit" })).toBeVisible();
});

test("mobile layout avoids horizontal overflow", async ({ page }) => {
  await page.goto("/?fixture=1");
  await page.waitForFunction(() => window.__USB_AGENTS_READY === true);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  expect(overflow).toBe(false);
  await expect(page).toHaveScreenshot("console-mobile-overflow.png", {
    fullPage: true,
    animations: "disabled"
  });
});
