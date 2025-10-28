import asyncio
from playwright.async_api import async_playwright
import sys

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream"
            ]
        )
        page = await browser.new_page()

        try:
            await page.goto("http://127.0.0.1:5000/pages/webcam.html")
            await page.wait_for_selector("#captureBtn:not([disabled])", state="visible", timeout=15000)
            await page.click("#captureBtn")
            await page.wait_for_selector("#ocrResult:not(:empty)", timeout=30000)
            await page.wait_for_selector("#translationResult:not(:empty)", timeout=30000)
            await page.screenshot(path="jules-scratch/verification/verification.png")
            print("Screenshot captured: jules-scratch/verification/verification.png")

        except Exception as e:
            print(f"An error occurred during verification: {e}", file=sys.stderr)
            await page.screenshot(path="jules-scratch/verification/verification_error.png")
            sys.exit(1)
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
