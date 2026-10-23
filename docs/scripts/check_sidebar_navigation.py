"""Exercise sidebar selection with browser caching and legacy session state."""

from __future__ import annotations

import argparse
import functools
import http.server
import re
import threading
import zlib
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "docs/_build/html")
    args = parser.parse_args()
    build = args.build_dir.resolve()
    content = (build / "_static/site-navigation.js").read_bytes().replace(b"\r", b"")
    expected = f"{zlib.crc32(content):08x}"
    for html in build.rglob("*.html"):
        for version in re.findall(r'site-navigation\.js\?v=([^"\s]+)', html.read_text()):
            assert version == expected, f"Stale navigation asset in {html}: {version}"
    handler = functools.partial(QuietHandler, directory=str(build))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    paths = ["gam.html", "gam_formula.html", "gam_smooths.html"]
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.route("https://**/*", lambda route: route.abort())
            page.goto(f"{origin}/api/gam.html", wait_until="domcontentloaded")
            sections = page.locator(".wy-menu-vertical p.caption").evaluate_all(
                "captions => captions.map(p => ({title: p.textContent.trim(), "
                "items: Array.from(p.nextElementSibling.children).map(li => "
                "li.querySelector(':scope > a')?.textContent.trim())}))"
            )
            assert "Tutorials" not in [section["title"] for section in sections]
            getting_started = next(
                section for section in sections if section["title"] == "Getting Started"
            )
            assert getting_started["items"] == [
                "Home", "Introductory examples", "Advanced examples", "Frequently Asked Questions"
            ], getting_started
            page.evaluate("""() => sessionStorage.setItem('nampy-sidebar-navigation',
                JSON.stringify({destination: location.pathname, scrollTop: 400,
                    expanded: ['/api/gam.html', '/api/gam_formula.html',
                               '/api/gam_smooths.html']}))""")

            def check(name):
                page.wait_for_function("document.querySelector('.wy-menu-vertical')")
                selected = page.locator(
                    ".wy-menu-vertical li.toctree-l1.current > a"
                ).evaluate_all("links => links.map(a => new URL(a.href).pathname)")
                assert selected == [f"/api/{name}"], selected
                # A new source must never be shipped under the old asset URL.
                src = page.locator('script[src*="site-navigation.js"]').get_attribute("src")
                assert f"v={expected}" in src, (src, expected)

            page.reload(wait_until="domcontentloaded")
            check(paths[0])
            for name in paths[1:] + paths + paths[1:]:
                page.locator(f'.wy-menu-vertical a[href="{name}"]').first.click()
                page.wait_for_url(f"**/api/{name}")
                page.wait_for_load_state("domcontentloaded")
                check(name)
                page.reload(wait_until="domcontentloaded")
                check(name)
            page.go_back(wait_until="domcontentloaded")
            check("gam_formula.html")
            page.go_forward(wait_until="domcontentloaded")
            check("gam_smooths.html")
            browser.close()
            print("PASS: repeated sidebar clicks, reloads, history, legacy state, and asset versions")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
