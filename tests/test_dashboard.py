"""Playwright-based browser tests for the Streamlit dashboard.

Launches the dashboard headless, navigates each page, screenshots, and
checks for visible Streamlit error/warning indicators.

Run:
    uv run python tests/test_dashboard.py
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCREENSHOT_DIR = ROOT / "tests" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

PAGES = [
    "📊 Overview",
    "🎯 Game Predictions",
    "💰 Betting Analysis",
    "📈 Track Record",
    "📝 Commentary",
    "📚 Historical Finals",
    "🔬 Methodology",
]


async def run_tests():
    from playwright.async_api import async_playwright

    results = {"pass": [], "fail": [], "warnings": []}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1400, "height": 900},
                                              device_scale_factor=2)
        page = await context.new_page()

        # Capture console errors
        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: console_errors.append(f"PAGE ERROR: {err}"))

        print("Loading dashboard...")
        await page.goto("http://localhost:8765", wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(3000)  # let plotly render

        for page_name in PAGES:
            print(f"\n--- Testing: {page_name} ---")
            console_errors.clear()
            try:
                # Click the radio button for this page
                await page.click(f"text={page_name}", timeout=10000)
                await page.wait_for_timeout(3500)  # plotly + data loads

                # Check for Streamlit error markers
                error_count = await page.locator("[data-baseweb='notification'][kind='error']").count()
                warning_count = await page.locator("[data-baseweb='notification'][kind='warning']").count()
                error_text = await page.locator(".stException").count()

                # Check for visible "Traceback" anywhere
                content = await page.content()
                has_traceback = "Traceback" in content and "<pre" in content

                # Screenshot
                safe_name = page_name.split(" ", 1)[1].lower().replace(" ", "_")
                screenshot_path = SCREENSHOT_DIR / f"{safe_name}.png"
                await page.screenshot(path=str(screenshot_path), full_page=True)

                if error_count > 0 or error_text > 0 or has_traceback:
                    results["fail"].append({
                        "page": page_name,
                        "errors": error_count + error_text,
                        "traceback": has_traceback,
                        "screenshot": str(screenshot_path),
                        "console": console_errors[:3],
                    })
                    print(f"  ✗ FAIL: {error_count + error_text} errors, traceback={has_traceback}")
                else:
                    results["pass"].append({"page": page_name, "screenshot": str(screenshot_path)})
                    print(f"  ✓ PASS — screenshot: {screenshot_path.name}")

                if warning_count > 0:
                    results["warnings"].append({"page": page_name, "count": warning_count})
                    print(f"  ⚠ {warning_count} warning(s) shown")

                if console_errors:
                    print(f"  Console errors: {console_errors[:2]}")
            except Exception as e:
                results["fail"].append({"page": page_name, "exception": str(e)})
                print(f"  ✗ EXCEPTION: {e}")

        await browser.close()

    print("\n" + "=" * 60)
    print(f"DASHBOARD TEST SUMMARY")
    print("=" * 60)
    print(f"  Passed: {len(results['pass'])}/{len(PAGES)}")
    print(f"  Failed: {len(results['fail'])}")
    print(f"  Warnings: {len(results['warnings'])}")
    if results["fail"]:
        print("\nFailures:")
        for f in results["fail"]:
            print(f"  • {f['page']}: {f}")

    return results


if __name__ == "__main__":
    results = asyncio.run(run_tests())
    sys.exit(0 if not results["fail"] else 1)
