import asyncio
from playwright.async_api import async_playwright, expect
import sys
import io

# Ensure stdout and stderr are UTF-8 encoded
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

async def run_test(page, test_name, input_text, expected_text):
    """Helper function to run a single translation test case."""
    print(f"--- Running test: {test_name} ---")
    try:
        # Clear the input field before typing
        await page.fill("#inputWord", "")
        await page.fill("#inputWord", input_text)
        print(f"Filled input with: '{input_text}'")

        await page.click("button.btn-translate")
        print("Clicked translate button")

        # Increased timeout to handle potential API delays
        await page.wait_for_selector("#translationOutput:not(:text-is('Translating...'))", timeout=20000)

        # Use Playwright's expect for robust assertion
        await expect(page.locator("#translationOutput")).to_have_text(expected_text)

        output_text = await page.inner_text("#translationOutput")
        print(f"Output received: '{output_text}'")
        print(f"--- PASSED: {test_name} ---")
        return True

    except Exception as e:
        print(f"--- FAILED: {test_name}: {e} ---")
        await page.screenshot(path=f"failed_test_{test_name}.png")
        print(f"Screenshot saved to failed_test_{test_name}.png")
        return False


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        all_tests_passed = True

        try:
            await page.goto("http://127.0.0.1:5000/pages/hale-hosa-kannada.html")
            print("Navigated to the translation page.")

            # Test 1: Single word from the dictionary (should work as before)
            if not await run_test(page, "single_word_dictionary", "ಅಂಕ", "ಗುರುತು ಪ್ರಸಿದ್ಧ ತೊಡೆ ಸನ್ನೆಕೋಲು ಯುದ್ಧ"):
                all_tests_passed = False

            # Test 2: Multi-word with one in dictionary and one not (API fails)
            # Expected: translation of "ಅಂಕ" + placeholder for "ಕೆಲವು"
            expected_hybrid_output = "ಗುರುತು ಪ್ರಸಿದ್ಧ ತೊಡೆ ಸನ್ನೆಕೋಲು ಯುದ್ಧ [ಕೆಲವು: unavailable]"
            if not await run_test(page, "multi_word_hybrid_api_fail", "ಅಂಕ ಕೆಲವು", expected_hybrid_output):
                all_tests_passed = False

            # Test 3: The user's original input (both words not in dictionary, API fails)
            # Expected: placeholder for both words
            expected_api_only_output = "[ಕೆಲವು: unavailable] [ಪದಗಳು: unavailable]"
            if not await run_test(page, "user_input_api_fail", "ಕೆಲವು ಪದಗಳು", expected_api_only_output):
                all_tests_passed = False

        finally:
            await browser.close()
            if not all_tests_passed:
                print("\nOne or more tests failed.")
                sys.exit(1)
            else:
                print("\nAll verification tests passed successfully.")

if __name__ == "__main__":
    asyncio.run(main())
