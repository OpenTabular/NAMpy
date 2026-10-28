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
        for version in re.findall(
            r'site-navigation\.js\?v=([^"\s]+)', html.read_text()
        ):
            assert version == expected, f"Stale navigation asset in {html}: {version}"
    handler = functools.partial(QuietHandler, directory=str(build))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    paths = ["index.html", "smooths/index.html", "families/index.html"]
    labels = {"index.html": "Statistical models", "smooths/index.html": "Smooth bases and tensor products", "families/index.html": "Response families"}
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(accept_downloads=True)
            page.route("https://**/*", lambda route: route.abort())
            page.goto(f"{origin}/statistical_reference/index.html", wait_until="domcontentloaded")
            sections = page.locator(".wy-menu-vertical p.caption").evaluate_all(
                "captions => captions.map(p => ({title: p.textContent.trim(), "
                "items: Array.from(p.nextElementSibling.children).map(li => "
                "li.querySelector(':scope > a')?.textContent.trim())}))"
            )
            titles = [section["title"] for section in sections]
            assert titles.index("Module code") < titles.index("Development")
            assert page.locator("#rtd-search-form").count() == 0
            fonts = page.evaluate("""() => {
                const properties = ['fontFamily', 'fontSize', 'fontWeight', 'lineHeight'];
                const read = element => properties.map(key => getComputedStyle(element)[key]);
                return {
                    reference: read(document.querySelector('.wy-menu-vertical .toctree-l1 > a')),
                    groups: Array.from(document.querySelectorAll('.module-code-navigation summary'), read)
                };
            }""")
            assert fonts["groups"] and all(font == fonts["reference"] for font in fonts["groups"]), fonts
            assert "Tutorials" not in [section["title"] for section in sections]
            development = next(section for section in sections if section["title"] == "Development")
            assert development["items"] == [
                "Contributing to NAMpy", "Architecture and validation", "Validation reports", "License"
            ], development
            reference = next(section for section in sections if section["title"] == "Reference")
            assert "Compatibility and limitations" in reference["items"], reference
            # Links in shared Markdown must not leave a local/versioned build.
            linked_page = browser.new_page()
            linked_page.route("https://**/*", lambda route: route.abort())
            linked_page.goto(f"{origin}/contributing.html", wait_until="domcontentloaded")

            def sidebar_outline():
                return linked_page.locator(".wy-menu-vertical p.caption").evaluate_all(
                    "nodes => nodes.map(n => ({title: n.textContent.trim(), "
                    "items: Array.from(n.nextElementSibling.children).map(li => "
                    "li.querySelector(':scope > a')?.textContent.trim())}))"
                )

            outline = sidebar_outline()
            for _ in range(2):
                linked_page.get_by_role("link", name="compatibility page", exact=True).click()
                linked_page.wait_for_url(f"{origin}/compatibility.html")
                assert sidebar_outline() == outline
                assert linked_page.locator("h1").inner_text().startswith("Compatibility")
                linked_page.reload(wait_until="domcontentloaded")
                assert sidebar_outline() == outline
                linked_page.go_back(wait_until="domcontentloaded")
            assert linked_page.locator('a[href*="reference_sources"]').count() == 0
            linked_page.close()
            getting_started = next(
                section for section in sections if section["title"] == "Getting Started"
            )
            assert getting_started["items"] == [
                "Home",
                "Tutorials",
                "Model guide",
                "Frequently Asked Questions",
                "Download documentation",
            ], getting_started
            page.get_by_role("link", name="Download documentation", exact=True).click()
            page.wait_for_url(f"{origin}/downloads.html")
            assert page.locator("h1").inner_text().startswith("Download documentation")
            assert page.locator(
                ".wy-menu-vertical li.current > a.current"
            ).evaluate_all("links => links.map(a => new URL(a.href).pathname)") == [
                "/downloads.html"
            ]
            assert page.get_by_role("link", name=re.compile(r"^Docs workflow")).get_attribute(
                "href"
            ) == "https://github.com/OpenTabular/NAMpy/actions/workflows/docs.yml"
            page.go_back(wait_until="domcontentloaded")
            assert "Concepts" not in [section["title"] for section in sections]
            for title, minimum in [("How-to Guides", 12)]:
                section = next(item for item in sections if item["title"] == title)
                assert title not in section["items"], section
                assert len(section["items"]) >= minimum, section
            reference = next(item for item in sections if item["title"] == "Reference")
            assert reference["items"] == [
                "Statistical models", "Neural models", "Predictions and explanations",
                "Plotting", "Extension APIs", "Compatibility and limitations",
            ], reference
            assert "Statistical Reference" not in [section["title"] for section in sections]
            assert "Neural Reference" not in [section["title"] for section in sections]
            page.evaluate("""() => sessionStorage.setItem('nampy-sidebar-navigation',
                JSON.stringify({destination: location.pathname, scrollTop: 400,
                    expanded: ['/api/gam.html', '/api/gam_formula.html',
                               '/api/gam_smooths.html']}))""")

            def check(name):
                page.wait_for_function("document.querySelector('.wy-menu-vertical')")
                selected = page.locator(
                    ".wy-menu-vertical li.current > a.current"
                ).evaluate_all("links => links.map(a => new URL(a.href).pathname)")
                # A nested page also marks its GAM ancestor; only its own link
                # receives the active-link class.
                assert selected == [f"/statistical_reference/{name}"], selected
                # A new source must never be shipped under the old asset URL.
                src = page.locator('script[src*="site-navigation.js"]').get_attribute(
                    "src"
                )
                assert f"v={expected}" in src, (src, expected)

            page.reload(wait_until="domcontentloaded")
            check(paths[0])
            for name in paths[1:] + paths + paths[1:]:
                page.locator(".wy-menu-vertical a:visible").filter(has_text=re.compile("^" + re.escape(labels[name]) + "$" )).first.click()
                page.wait_for_url(f"**/statistical_reference/{name}")
                page.wait_for_load_state("domcontentloaded")
                check(name)
                page.reload(wait_until="domcontentloaded")
                check(name)
            page.go_back(wait_until="domcontentloaded")
            check("smooths/index.html")
            page.go_forward(wait_until="domcontentloaded")
            check("families/index.html")
            # Existing runnable examples must enter the same central hierarchy.
            page.goto(f"{origin}/examples/fit_models.html", wait_until="domcontentloaded")
            page.locator("a.api-code-link").filter(has_text=re.compile("^NAMRegressor$")).first.click()
            page.wait_for_url(f"{origin}/neural_reference/estimators/NAMRegressor.html")
            active = page.locator(".wy-menu-vertical li.current > a.current")
            assert active.evaluate_all("links => links.map(a => new URL(a.href).pathname)") == ["/neural_reference/estimators/NAMRegressor.html"]
            assert active.is_visible()
            # The formula API has one parent even though its old URL is retained.
            page.goto(f"{origin}/api/gam_formula.html", wait_until="domcontentloaded")
            assert page.locator(".wy-menu-vertical li.current > a.current").count() == 1
            page.goto(f"{origin}/tutorials/index.html", wait_until="domcontentloaded")
            expected_tutorials = [
                "notebooks/01_getting_started.html",
                "notebooks/05_real_data_classification.html",
                "tutorials/first_gam.html",
                "notebooks/group_smooths.html",
                "notebooks/smoothness_selection.html",
                "notebooks/derivatives_contrasts.html",
                "notebooks/gam_diagnostics.html",
                "notebooks/tensor_interactions.html",
                "tutorials/gam_nam.html",
                "notebooks/03_neural_models.html",
                "tutorials/interaction_selection.html",
                "notebooks/02_statistical_gams.html",
                "notebooks/overdispersed_counts.html",
                "notebooks/gamma_response.html",
                "notebooks/beta_response.html",
                "notebooks/tweedie_response.html",
                "notebooks/ordered_response.html",
                "notebooks/gamma_location_scale.html",
                "notebooks/04_distributional_models.html",
                "in_depth/quantile_calibration.html",
                "in_depth/tree_effects.html",
                "notebooks/regional_smoothing.html",
                "notebooks/correlated_errors.html",
                "notebooks/neighborhood_cv.html",
                "notebooks/adaptive_smoothing.html",
                "notebooks/spherical_smoothing.html",
                "notebooks/functional_inputs.html",
                "notebooks/curvature_constraints.html",
            ]
            card_targets = page.locator(".sd-card a.sd-stretched-link").evaluate_all(
                "links => links.map(a => new URL(a.href).pathname.slice(1))"
            )
            assert card_targets == expected_tutorials, card_targets
            downloads = page.locator(".sd-card a.reference.download")
            assert downloads.count() == len(expected_tutorials)
            for download in downloads.all():
                # Reset the document between clicks to avoid Chromium's
                # bulk-download limiter.
                page.reload(wait_until="domcontentloaded")
                with page.expect_download() as event:
                    download.click()
                assert event.value.suggested_filename.endswith(".ipynb")
            page.goto(f"{origin}/in_depth/index.html", wait_until="domcontentloaded")
            page.wait_for_url(f"{origin}/tutorials/index.html")
            browser.close()
            print(
                "PASS: repeated sidebar clicks, reloads, history, legacy state, and asset versions"
            )
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
