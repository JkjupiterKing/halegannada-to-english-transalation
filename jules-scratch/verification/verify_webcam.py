from playwright.sync_api import sync_playwright, expect
import time

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    try:
        # Give the server more time to start
        time.sleep(15)

        # Navigate to the webcam page
        page.goto("http://localhost:5000/pages/webcam.html", timeout=60000)

        # Wait for the capture button to be available and click it
        capture_button = page.locator("#captureBtn")
        expect(capture_button).to_be_enabled(timeout=10000)
        capture_button.click()

        # Wait for the results to appear
        expect(page.locator("#ocrResult")).not_to_be_empty(timeout=20000)
        expect(page.locator("#translationResult")).not_to_be_empty(timeout=20000)

        # Take a screenshot
        page.screenshot(path="jules-scratch/verification/verification.png")

        print("Verification script completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        page.screenshot(path="jules-scratch/verification/error.png")

    finally:
        # Clean up
        context.close()
        browser.close()

with sync_playwright() as playwright:
    run(playwright)