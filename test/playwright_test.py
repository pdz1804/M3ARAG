from playwright.sync_api import sync_playwright
import os

def sanitize_filename(url: str):
    """Sanitize domain name to create safe filename"""
    domain = url.split("//")[1].split("/")[0].replace(".", "_")
    return domain

def capture_screenshot_and_pdf(url: str, output_dir: str = "outputs", full_page: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    domain_safe = sanitize_filename(url)

    screenshot_path = os.path.join(output_dir, f"{domain_safe}.png")
    pdf_path = os.path.join(output_dir, f"{domain_safe}.pdf")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Load page and wait for it
        try:
            page.goto(url, wait_until="load", timeout=60000)
        except Exception as e:
            print(f"[✖] Failed to load {url}: {e}")
            browser.close()
            return False, False  # screenshot_failed, pdf_failed

        # Save full-page screenshot
        screenshot_success = False
        try:
            page.screenshot(path=screenshot_path, full_page=full_page)
            print(f"[✔] Screenshot saved to {screenshot_path}")
            screenshot_success = True
        except Exception as e:
            print(f"[✖] Screenshot failed for {url}: {e}")

        # Save page as PDF
        pdf_success = False
        try:
            page.pdf(path=pdf_path, format="A4")
            print(f"[✔] PDF saved to {pdf_path}")
            pdf_success = True
        except Exception as e:
            print(f"[✖] PDF generation failed for {url}: {e}")

        browser.close()
        return screenshot_success, pdf_success


# === Main Execution ===
if __name__ == "__main__":
    tech_urls = [
        "https://www.hagerty.ca/",
        "https://www.hagerty.com/media/car-profiles/when-macgyver-raced-a-camaro/",
        "https://www.txopartners.com/",
        "https://www.hagerty.com/fueling-car-culture",
        "https://www.nocera.company/service",
        "https://inovio.com/inovio-publications/",
        "https://www.invesco.com/jp/ja/individual-investor/funds/featured-funds/american-dynamic.html"
    ]

    total = len(tech_urls)
    shot_success = 0
    pdf_success = 0
    fail_count = 0

    for url in tech_urls:
        screenshot_ok, pdf_ok = capture_screenshot_and_pdf(url, output_dir="captures")
        if screenshot_ok:
            shot_success += 1
        if pdf_ok:
            pdf_success += 1
        if not screenshot_ok and not pdf_ok:
            fail_count += 1

    print("\n=== Summary ===")
    print(f"Total URLs processed   : {total}")
    print(f"✔ Successful screenshots: {shot_success}")
    print(f"✔ Successful PDFs       : {pdf_success}")
    print(f"✖ Total failed URLs     : {fail_count}")
