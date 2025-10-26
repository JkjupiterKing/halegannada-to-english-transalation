import { test, expect } from '@playwright/test';

test('hale-hosa-kannada translation test', async ({ page }) => {
  await page.goto('http://localhost:5000/pages/hale-hosa-kannada.html');
  await page.getByPlaceholder('Enter a word...').click();
  await page.getByPlaceholder('Enter a word...').fill('ಮರಮರ');
  await page.getByRole('button', { name: 'Translate' }).click();

  // Wait for the translation to appear in the correct element
  await page.waitForSelector('#translationOutput:has-text("ಮರ")');
});