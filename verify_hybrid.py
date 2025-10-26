
import asyncio
from playwright.async_api import async_playwright, expect

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto("http://127.0.0.1:5000/pages/hale-hosa-kannada.html")

            # Input a sentence with a known dictionary word ("ಅಂಕ") and a non-dictionary word ("ಪರೀಕ್ಷೆ")
            input_text = "ಅಂಕ ಪರೀಕ್ಷೆ"
            await page.locator("#text-input").fill(input_text)
            await page.locator("button:has-text('Translate')").click()

            # Wait for the translation to appear
            output_locator = page.locator("#kannada-output")
            await expect(output_locator).not_to_have_text(input_text, timeout=20000) # Wait up to 20s

            # Assert that the known word is translated and the other word is also handled
            translated_text = await output_locator.input_value()

            print(f"Input: '{input_text}'")
            print(f"Output: '{translated_text}'")

            # Check if the dictionary word "ಅಂಕ" is translated to "ಅಂಕ/ಯುದ್ಧ"
            # And that the second word is not the original word
            assert "ಅಂಕ/ಯುದ್ಧ" in translated_text
            assert "ಪರೀಕ್ಷೆ" not in translated_text

            print("Verification successful: Hybrid translation is working correctly.")

        except Exception as e:
            print(f"An error occurred: {e}")
            await page.screenshot(path="error_screenshot.png")
            print("Screenshot saved to error_screenshot.png")
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
