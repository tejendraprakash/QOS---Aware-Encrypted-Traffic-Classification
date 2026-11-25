import time
from pathlib import Path
from playwright.sync_api import sync_playwright

def inspect_dashboard():
    screenshot_dir = Path("dashboard/screenshots")
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1050})
        page = context.new_page()
        
        print("Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", timeout=60000)
        
        # Wait for Streamlit app container
        try:
            page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=30000)
        except Exception:
            print("stAppViewContainer not found, waiting for body")
            page.wait_for_selector("body", timeout=10000)
            
        print("App container loaded. Waiting 6 seconds for data & charts to render...")
        time.sleep(6)
        
        # 1. Capture Overview Page
        overview_path = screenshot_dir / "01_overview.png"
        page.screenshot(path=str(overview_path), full_page=True)
        print(f"Captured: {overview_path}")
        
        title = page.title()
        print(f"Page Title: {title}")
        
        def click_page(tab_text, screenshot_name):
            try:
                locator = page.locator(f"label:has-text('{tab_text}')").first
                if locator.is_visible():
                    locator.click()
                    print(f"Clicked radio option: {tab_text}")
                else:
                    alt = page.locator(f"text='{tab_text}'").first
                    alt.click()
                    print(f"Clicked text: {tab_text}")
                time.sleep(5)
                shot_path = screenshot_dir / screenshot_name
                page.screenshot(path=str(shot_path), full_page=True)
                print(f"Captured: {shot_path}")
            except Exception as e:
                print(f"Failed to navigate to {tab_text}: {e}")

        # 2. Traffic Classification
        click_page("Traffic Classification", "02_traffic_classification.png")
        
        # 3. Model Performance
        click_page("Model Performance", "03_model_performance.png")
        
        # 4. QoS Performance
        click_page("QoS Performance", "04_qos_performance.png")
        
        browser.close()
        print("Playwright UI audit complete!")

if __name__ == "__main__":
    inspect_dashboard()
