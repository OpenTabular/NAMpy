"""Execute the documented practical workflows and save their displayed results."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from generate_detailed_assets import csv_table, save

ROOT = Path(__file__).resolve().parents[2]


def execute(path, after=None):
    source = path.read_text()
    if after:
        source = source.split(after, 1)[1]
    namespace = {}
    for block in re.findall(
        r"^\.\. code-block:: python\n\n((?:(?:   [^\n]*)?\n)+)", source, re.M
    ):
        code = "\n".join(
            line[3:] if line.startswith("   ") else line for line in block.splitlines()
        )
        exec(compile(code, str(path), "exec"), namespace)
    return namespace


def main():
    result = execute(
        ROOT / "docs/howto/offsets.rst",
        "Neural binary-classification offsets at inference",
    )
    csv_table(
        "neural_offset_results",
        "Offset inference on untouched test rows",
        result["offset_results"],
    )
    selected = execute(ROOT / "docs/howto/interaction_selection.rst")
    csv_table(
        "interaction_selection_results",
        "Final test performance after training/validation selection",
        selected["interaction_results"],
    )
    csv_table(
        "interaction_detector",
        "Training/validation detector decisions",
        selected["selection_table"],
    )
    save(selected["figure"], "interaction_selection_workflow")
    plt.close("all")
    print("PASS: documented neural offset and interaction-selection workflows")


if __name__ == "__main__":
    main()
