"""Notebook links to folded guides must resolve inside the built manual."""

import json
import sys
from pathlib import Path

from sphinx.cmd.build import build_main

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
sys.path.insert(0, str(DOCS / "_ext"))

from howto_groups import FOLDED_DOCNAMES, folded_guide_target  # noqa: E402


def _notebook_links():
    links = []
    for path in sorted((DOCS / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text())
        docname = "notebooks/" + path.stem
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "markdown":
                continue
            text = "".join(cell["source"])
            start = 0
            while True:
                found = text.find("](", start)
                if found < 0:
                    break
                end = text.find(")", found + 2)
                target = text[found + 2 : end]
                start = end + 1
                resolved = folded_guide_target(docname, target)
                if resolved is not None or any(
                    target.endswith(name + ".rst") or target.endswith("/" + name + ".rst")
                    for name in FOLDED_DOCNAMES
                ):
                    links.append((path.name, target, resolved))
    return links


def test_notebook_links_to_folded_guides_use_howto_anchors():
    links = _notebook_links()
    assert links
    assert {target for _, target, _ in links} == {
        "../user_guide/calibration.rst",
        "../user_guide/formula_visual_reference.rst",
        "../user_guide/quantile_inference.rst",
        "../user_guide/semistructured.rst",
        "../user_guide/survival.rst",
    }
    assert all(resolved and resolved.startswith("howto-") for _, _, resolved in links)


def test_other_local_links_stay_untouched():
    assert folded_guide_target("notebooks/demo", "../howto/calibrate.rst") is None
    assert folded_guide_target("notebooks/demo", "https://example.com/a.rst") is None
    assert folded_guide_target("notebooks/demo", "#section") is None


def test_epub_does_not_package_folded_guide_sources(tmp_path):
    source = tmp_path / "src"
    (source / "user_guide").mkdir(parents=True)
    (source / "howto").mkdir()
    (source / "notebooks").mkdir()
    (source / "user_guide" / "calibration.rst").write_text(
        "Calibrate scalar intervals\n"
        "==========================\n"
        "\n"
        "Hold the calibration rows out of fitting.\n"
    )
    (source / "howto" / "calibrate.rst").write_text(
        "Calibrate predictions\n"
        "=====================\n"
        "\n"
        ".. howto-dropdown:: user_guide/calibration.rst\n"
    )
    (source / "notebooks" / "demo.ipynb").write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3",
                    }
                },
                "cells": [
                    {
                        "id": "demo",
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [
                            "# Demo\n",
                            "\n",
                            "See [calibration](../user_guide/calibration.rst).\n",
                        ],
                    }
                ],
            }
        )
    )
    (source / "index.rst").write_text(
        "Manual\n"
        "======\n"
        "\n"
        ".. toctree::\n"
        "\n"
        "   howto/calibrate\n"
        "   notebooks/demo\n"
    )
    (source / "conf.py").write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(DOCS / '_ext')!r})\n"
        "project = 'NAMpy'\n"
        "copyright = '2026, NAMpy'\n"
        "version = '0'\n"
        "extensions = ['howto_groups', 'sphinx_design', 'nbsphinx']\n"
        "nbsphinx_execute = 'never'\n"
        "exclude_patterns = ['user_guide/calibration.rst']\n"
        "epub_exclude_files = ['notebooks/demo.ipynb']\n"
        "master_doc = 'index'\n"
    )
    output = tmp_path / "epub"
    status = build_main(
        [
            "-W",
            "--keep-going",
            "-b",
            "epub",
            "-d",
            str(tmp_path / "doctrees"),
            str(source),
            str(output),
        ]
    )
    assert status == 0
    assert not (output / "user_guide" / "calibration.rst").exists()
    page = (output / "notebooks" / "demo.xhtml").read_text()
    assert "howto-user-guide-calibration" in page
    assert "calibration.rst" not in page
