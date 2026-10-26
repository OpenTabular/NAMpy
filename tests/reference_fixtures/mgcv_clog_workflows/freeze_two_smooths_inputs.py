"""Record the original Linux input for the existing two-smooth R captures.

This preserves the historical input generator, not a portable way to recreate
identical floats: NumPy's cosine can differ in its last bits across platforms.
Normal tests read the committed JSON and never execute this script.
"""

import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parent
    inputs = json.loads((root.parent / "mgcv_cnorm_workflows" / "inputs.json").read_text())
    frozen = {"z": np.cos(np.arange(len(inputs["y"])) * 0.731).tolist()}
    output = root / "two_smooths_inputs.json"
    if output.exists() and json.loads(output.read_text()) != frozen:
        raise RuntimeError("This platform differs from the frozen source-capture input.")
    output.write_text(json.dumps(frozen, indent=2) + "\n")


if __name__ == "__main__":
    main()
