from playwright.sync_api import sync_playwright, expect

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    # Verify text-translation.html
    page.goto("http://127.0.0.1:5000/pages/text-translation.html")
    page.screenshot(path="jules-scratch/verification/01-text-translation-initial.png")

    page.locator("#inputText").fill("ನಮಸ್ಕಾರ")
    page.locator("button[onclick='speakHalegannada(this)']").click()
    page.wait_for_timeout(1000) # Wait for speech to start
    page.screenshot(path="jules-scratch/verification/02-text-translation-speaking-input.png")

    page.locator("button[onclick='translateText()']").click()
    expect(page.locator("#englishOutput")).not_to_be_empty()
    page.locator("button[onclick='speakEnglish(this)']").click()
    page.wait_for_timeout(1000)
    page.screenshot(path="jules-scratch/verification/03-text-translation-speaking-english.png")

    # Verify hale-osa-kannada.html
    page.goto("http://127.0.0.1:5000/pages/hale-osa-kannada.html")
    page.screenshot(path="jules-scratch/verification/04-hale-osa-kannada-initial.png")

    page.locator("#inputWord").fill("ಪದ")
    page.locator("button[onclick='speakHalegannada(this)']").click()
    page.wait_for_timeout(1000)
    page.screenshot(path="jules-scratch/verification/05-hale-osa-kannada-speaking-input.png")

    page.locator("button[onclick='translateWord()']").click()
    expect(page.locator("#translationOutput")).not_to_be_empty()
    page.locator("button[onclick='speakTranslation(this)']").click()
    page.wait_for_timeout(1000)
    page.screenshot(path="jules-scratch/verification/06-hale-osa-kannada-speaking-translation.png")

    # Verify hale-hosa-seq2seq.html
    page.goto("http://127.0.0.1:5000/pages/hale-hosa-seq2seq.html")
    page.screenshot(path="jules-scratch/verification/07-hale-hosa-seq2seq-initial.png")

    page.locator("#inputText").fill("ಹಳೆಗನ್ನಡ")
    page.locator("button[onclick='speakHalegannada(this)']").click()
    page.wait_for_timeout(1000)
    page.screenshot(path="jules-scratch/verification/08-hale-hosa-seq2seq-speaking-input.png")

    page.locator("button[onclick='translateText()']").click()
    expect(page.locator("#englishOutput")).not_to_be_empty()
    page.locator("button[onclick='speakEnglish(this)']").click()
    page.wait_for_timeout(1000)
    page.screenshot(path="jules-scratch/verification/09-hale-hosa-seq2seq-speaking-english.png")


    browser.close()

with sync_playwright() as playwright:
    run(playwright)