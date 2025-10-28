import asyncio
from playwright.async_api import async_playwright, expect

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        try:
            # Grant camera permissions
            await page.context.grant_permissions(['camera'])

            # Test Webcam Translation page
            await page.goto('http://127.0.0.1:5000/pages/webcam.html', timeout=60000)

            # Wait for the video element to be ready
            await page.wait_for_selector('video#webcam', timeout=10000)

            print("Webcam page loaded. Clicking capture button...")
            await page.click('button#captureBtn')

            # Wait for the OCR result to appear, which indicates server response
            # The Gemini API is mocked, but the server logic still runs.
            # The mock should return a predictable response.
            ocr_result_selector = '#ocrResult'
            await expect(page.locator(ocr_result_selector)).not_to_be_empty(timeout=30000)

            extracted_text = await page.inner_text(ocr_result_selector)
            print(f"Extracted Text: {extracted_text}")

            translation_result_selector = '#translationResult'
            await expect(page.locator(translation_result_selector)).not_to_be_empty(timeout=30000)

            translation_text = await page.inner_text(translation_result_selector)
            print(f"Translation: {translation_text}")

            # The Gemini OCR is mocked on the server side, so we can't check for specific text,
            # but we can check that the fields are populated.
            if extracted_text and translation_text:
                print("SUCCESS: Webcam verification complete. OCR and translation fields are populated.")
            else:
                print("FAILURE: Webcam verification failed. OCR or translation fields are empty.")

        except Exception as e:
            print(f"An error occurred during webcam verification: {e}")
            await page.screenshot(path='jules-scratch/webcam_error.png')
            print("Screenshot saved to jules-scratch/webcam_error.png")

        finally:
            await browser.close()

if __name__ == '__main__':
    asyncio.run(main())