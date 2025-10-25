
from playwright.sync_api import sync_playwright, expect

def run(playwright):
    browser = playwright.chromium.launch()
    page = browser.new_page()

    # Verify text-translation page
    page.goto("http://127.0.0.1:5000/pages/text-translation.html")
    page.locator("#inputText").fill("ಟೆಸ್ಟ್")
    page.locator("text=Translate").click()
    expect(page.locator("#kannadaOutput")).to_have_text("ಟೆಸ್ಟ್")
    page.screenshot(path="jules-scratch/verification/text-translation.png")

    # Verify hale-osa-kannada page
    page.goto("http://127.0.0.1:5000/pages/hale-osa-kannada.html")
    page.locator("#inputWord").fill("ಟೆಸ್ಟ್")
    page.locator("text=Translate").click()
    expect(page.locator("#translationOutput")).to_have_text("ಟೆಸ್ಟ್")
    page.screenshot(path="jules-scratch/verification/hale-osa-kannada.png")

    # Verify hale-hosa-seq2seq page
    page.goto("http://127.0.0.1:5000/pages/hale-hosa-seq2seq.html")
    page.locator("#inputText").fill("ಟೆಸ್ಟ್")
    page.locator("text=Translate").click()
    expect(page.locator("#kannadaOutput")).to_have_text("ಟೆಸ್ಟ್")
    page.screenshot(path="jules-scratch/verification/hale-hosa-seq2seq.png")

    browser.close()

with sync_playwright() as playwright:
    run(playwright)
