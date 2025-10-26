import { test, expect } from '@playwright/test';

test('test', async ({ page }) => {
  await page.goto('http://localhost:5000/pages/text-translation.html');
  await page.getByPlaceholder('Enter your text here...').click();
  await page.getByPlaceholder('Enter your text here...').fill('ಮರಮರ');
  await page.getByRole('button', { name: 'Translate' }).click();

  // Wait for the translation to appear
  await page.waitForSelector('#kannadaOutput:has-text("ಮರ")');

  await page.screenshot({ path: 'screenshot.png' });
});