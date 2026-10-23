"""Save the plot from the homepage's exact displayed example."""

import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = (ROOT / "docs/index.rst").read_text()
    block = re.search(
        r"^\.\. code-block:: python\n\n((?:(?:   [^\n]*)?\n)+)", source, re.M
    )[1]
    code = "\n".join(
        line[3:] if line.startswith("   ") else line for line in block.splitlines()
    )
    namespace = {}
    exec(compile(code, "<homepage>", "exec"), namespace)
    figures = namespace.get("figures") or namespace["model"].plot(pages=1)
    figure = figures["figures"][0]
    figure.set_size_inches(6, 4)
    figure.tight_layout()
    figure.savefig(
        ROOT / "docs/_static/home_quickstart.png", dpi=180, bbox_inches="tight"
    )
    plt.close("all")


if __name__ == "__main__":
    main()
