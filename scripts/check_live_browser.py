"""Optional real-browser smoke of a running viewer; runs ONE bounded demo round.

Uses an isolated Playwright interpreter; not a runtime dependency of the viewer.
"""

import argparse
import json
from pathlib import Path
import time

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8766")
    parser.add_argument("--output", default="artifacts/live_validation/browser")
    parser.add_argument("--gpus", type=int, default=4)
    args = parser.parse_args()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    checks = {}
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1280})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until="domcontentloaded")
        page.wait_for_function("!document.getElementById('run').disabled", timeout=240000)
        assert page.locator("article").count() == args.gpus
        page.wait_for_function(
            "Array.from(document.querySelectorAll('article img')).every(i=>i.complete&&i.naturalWidth>0)",
            timeout=30000,
        )
        checks["all_actual_camera_images_loaded"] = True
        page.locator("#run").click()
        page.wait_for_function("!document.getElementById('pause').disabled", timeout=30000)
        page.locator("#pause").click()
        page.wait_for_function(
            "Array.from(document.querySelectorAll('.phase')).every(p=>p.textContent==='paused')",
            timeout=30000,
        )
        paused = page.request.get(args.url + "/api/status").json()
        checks["browser_start_and_pause_all_gpus"] = True
        page.locator("#view").select_option("top")
        time.sleep(2)
        current = page.request.get(args.url + "/api/status").json()
        assert current["control"]["view"] == "top"
        assert [j["observation"]["time"] for j in paused["jobs"]] == [
            j["observation"]["time"] for j in current["jobs"]
        ]
        checks["browser_camera_switch_preserves_paused_physics"] = True
        page.screenshot(path=str(directory / "paused_top.png"), full_page=True)
        hero = page.locator("article").first.locator(".world")
        bounds = hero.bounding_box()
        page.mouse.move(bounds["x"] + bounds["width"] * 0.4, bounds["y"] + bounds["height"] * 0.5)
        page.mouse.down()
        page.mouse.move(
            bounds["x"] + bounds["width"] * 0.5, bounds["y"] + bounds["height"] * 0.6, steps=8
        )
        page.mouse.up()
        page.wait_for_function("document.getElementById('view').value==='orbit'", timeout=10000)
        control = page.request.get(args.url + "/api/status").json()["control"]
        assert control["view"] == "orbit"
        assert control["azimuth"] != -55
        checks["mouse_drag_orbits_native_camera"] = True
        page.mouse.wheel(0, -120)
        time.sleep(1)
        assert (
            page.request.get(args.url + "/api/status").json()["control"]["distance"]
            < control["distance"]
        )
        checks["mouse_wheel_zooms_native_camera"] = True
        page.locator("article").first.locator(".fullscreen").click()
        page.wait_for_function("document.fullscreenElement!==null", timeout=10000)
        page.evaluate("document.exitFullscreen()")
        checks["fullscreen_works"] = True
        page.locator("#view").select_option("overview")
        page.locator("#pause").click()
        page.wait_for_function(
            "Array.from(document.querySelectorAll('.phase')).every(p=>p.textContent==='done')",
            timeout=180000,
        )
        checks["browser_resume_and_bounded_missions_finish"] = True
        page.wait_for_function("!document.getElementById('run').disabled", timeout=10000)
        checks["replay_button_ready"] = True
        assert not errors, errors
        checks["no_javascript_errors"] = True
        page.screenshot(path=str(directory / "completed_overview.png"), full_page=True)
        final = page.request.get(args.url + "/api/status").json()
        browser.close()
    (directory / "checks.json").write_text(
        json.dumps(
            {
                "passed": all(checks.values()),
                "checks": checks,
                "javascript_errors": errors,
                "final_status": final,
                "full_pilot_executed": False,
            },
            indent=2,
        )
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
