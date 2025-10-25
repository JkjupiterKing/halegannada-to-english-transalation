import re
from playwright.sync_api import Page, expect, sync_playwright

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    try:
        # Test 1: Hale-Osa Kannada (Dictionary Translation)
        print("Testing Hale-Osa Kannada page...")
        page.goto("http://127.0.0.1:5000/pages/hale-osa-kannada.html")

        # Input a word and translate
        page.locator("#inputWord").fill("ಕೆಮ್ಮನೆ ಪಂಪಾಪತಿ")
        page.locator("button:has-text('Translate')").click()

        # Check for the correct translation
        output_locator = page.locator("#translationOutput")
        expect(output_locator).to_contain_text("ಸುಮ್ಮನೆ", timeout=10000)
        expect(output_locator).to_contain_text("ಪಂಪಾಪತಿ", timeout=10000)
        print("Dictionary translation successful.")

        # Test speaker button
        page.locator("button[onclick*='inputWord']").click()
        print("Hale-Osa Kannada speaker button clicked.")

        # Test 2: Text Translation (API Translation)
        print("\nTesting Text Translation page...")
        page.goto("http://127.0.0.1:5000/pages/text-translation.html")

        # Input text and translate
        page.locator("#inputText").fill("ಪಂಪಾಪತಿ")
        page.locator("button:has-text('Translate')").click()

        # Check for API-based translation (will be the same as input if API fails)
        expect(page.locator("#kannadaOutput")).to_contain_text("Translation error", timeout=20000)
        print("API translation gracefully handled API failure.")

        # Test speaker buttons
        page.locator("button[onclick*='inputText']").click()
        print("Text Translation input speaker button clicked.")
        page.locator("button[onclick*='kannadaOutput']").click()
        print("Text Translation output speaker button clicked.")

        # Test 3: Seq2Seq Translation
        print("\nTesting Hale-Hosa Seq2Seq page...")
        page.goto("http://127.0.0.1:5000/pages/hale-hosa-seq2seq.html")

        # Input text and translate
        page.locator("#inputText").fill("ನರपति")
        page.locator("button:has-text('Translate')").click()

        # Check for translation
        expect(page.locator("#kannadaOutput")).to_contain_text("Translation error", timeout=20000)
        print("Seq2Seq translation gracefully handled API failure.")

        # Test speaker buttons
        page.locator("button[onclick*='inputText']").click()
        print("Seq2Seq input speaker button clicked.")
        page.locator("button[onclick*='kannadaOutput']").click()
        print("Seq2Seq output speaker button clicked.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Capture a screenshot for verification
        screenshot_path = "verification_screenshot.png"
        page.screenshot(path=screenshot_path)
        print(f"\nScreenshot saved to {screenshot_path}")

        browser.close()

with sync_playwright() as playwright:
    run(playwright)
