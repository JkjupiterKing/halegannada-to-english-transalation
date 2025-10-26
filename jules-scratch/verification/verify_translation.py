from playwright.sync_api import sync_playwright, expect

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()

    # Navigate to the text translation page
    page.goto("http://localhost:5000/pages/text-translation.html")

    # Input Halegannada text
    page.locator("#inputText").fill("ಮರಮರ")

    # Click the translate button
    page.locator("button:has-text('Translate')").click()

    # Wait for the "Translating..." message to disappear
    expect(page.locator("#kannadaOutput")).not_to_have_text("Translating...", timeout=60000)

    # Take a screenshot
    page.screenshot(path="jules-scratch/verification/translation_verification.png")

    browser.close()

with sync_playwright() as playwright:
    run(playwright)