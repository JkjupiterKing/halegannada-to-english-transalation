
import asyncio
from playwright.async_api import async_playwright, expect

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto("http://127.0.0.1:5000/pages/webcam.html")
            await page.wait_for_selector("#captureBtn")
            await page.click("#captureBtn")
            await asyncio.sleep(5) # Wait for processing
            await page.screenshot(path="jules-scratch/verification/webcam_verification.png")
            print("Screenshot taken.")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            await browser.close()

asyncio.run(main())
