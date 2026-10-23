"""Check built local links, anchors, redirects, navigation, and notebook records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import nbformat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs/_ext"))
from site_integrity import REDIRECTS  # noqa: E402


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.links = []
        self.robots = ""
        self.redirect = None
        self.nav_labels = []
        self.stack = []
        self.anchor = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag in {"a", "link"} and attrs.get("href"):
            self.links.append(attrs["href"])
        if tag in {"img", "script"} and attrs.get("src"):
            self.links.append(attrs["src"])
        if tag == "meta":
            if attrs.get("name") == "robots":
                self.robots = attrs.get("content", "")
            if attrs.get("http-equiv", "").lower() == "refresh":
                self.redirect = attrs.get("content", "").split("url=", 1)[-1]
        if tag not in {
            "meta",
            "link",
            "img",
            "input",
            "br",
            "hr",
            "source",
            "wbr",
            "embed",
            "area",
            "base",
            "col",
            "param",
            "track",
        }:
            self.stack.append((tag, attrs.get("class", "")))
        if tag == "a" and any(
            "wy-menu-vertical" in cls.split() for _, cls in self.stack
        ):
            self.anchor = ""

    def handle_endtag(self, tag):
        if tag == "a" and self.anchor is not None:
            self.nav_labels.append(self.anchor.strip())
            self.anchor = None
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        if self.anchor is not None:
            self.anchor += data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "docs/_build/html")
    args = parser.parse_args()
    build = args.build_dir.resolve()
    pages = {p.resolve(): Page(p.read_text()) for p in build.rglob("*.html")}
    errors = []
    for path, page in pages.items():
        if any(re.match(r"^\d+\.\s", label) for label in page.nav_labels):
            errors.append(f"{path}: numbered tutorial navigation")
        for href in page.links + ([page.redirect] if page.redirect else []):
            url = urlsplit(href)
            if url.scheme or url.netloc:
                continue
            target = (
                (
                    build / unquote(url.path.lstrip("/"))
                    if url.path.startswith("/")
                    else path.parent / unquote(url.path)
                ).resolve()
                if url.path
                else path
            )
            if target.is_dir():
                target = target / "index.html"
            if not target.exists():
                errors.append(f"{path.relative_to(build)}: missing {href}")
                continue
            if (
                url.fragment
                and target in pages
                and unquote(url.fragment) not in pages[target].ids
            ):
                errors.append(f"{path.relative_to(build)}: missing anchor {href}")
    for path, page in pages.items():
        if any(label.startswith("nampy.") for label in page.nav_labels):
            errors.append(
                f"{path.relative_to(build)}: generated symbols leaked into global navigation"
            )
    homepage = pages.get((build / "index.html").resolve())
    if homepage is not None:
        for path, page in pages.items():
            if page.nav_labels and page.nav_labels != homepage.nav_labels:
                errors.append(
                    f"{path.relative_to(build)}: global navigation differs from the homepage"
                )
    for old in REDIRECTS:
        path = (build / (old + ".html")).resolve()
        page = pages.get(path)
        if page is None or not page.redirect or "noindex" not in page.robots:
            errors.append(f"{old}: redirect missing or indexable")
    search_source = (build / "searchindex.js").read_text().strip()
    search = json.loads(search_source.removeprefix("Search.setIndex(").rstrip(";)"))
    for old in REDIRECTS:
        if old in search["docnames"]:
            errors.append(f"{old}: obsolete page remains indexed")
    for path in [
        *(ROOT / "docs/notebooks").glob("*.ipynb"),
        *(ROOT / "docs/development/notebooks").glob("*.ipynb"),
    ]:
        notebook = nbformat.read(path, as_version=4)
        if "nampy_execution" not in notebook.metadata:
            errors.append(f"{path.name}: missing execution provenance")
        fingerprint = hashlib.sha256(
            json.dumps(
                [cell.source for cell in notebook.cells if cell.cell_type == "code"],
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        if (
            notebook.metadata.get("nampy_execution", {}).get("code_sha256")
            != fingerprint
        ):
            errors.append(
                f"{path.name}: code changed since saved execution; rerun the notebook"
            )
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            if cell.execution_count is None:
                errors.append(f"{path.name}: unexecuted cell")
            if any(output.output_type == "error" for output in cell.outputs):
                errors.append(f"{path.name}: saved execution error")
    if errors:
        print("\n".join(errors[:100]))
        raise SystemExit(f"{len(errors)} documentation check failures")
    print(
        f"PASS: {len(pages)} HTML pages; links, anchors, redirects, navigation, and notebook records."
    )


if __name__ == "__main__":
    main()
