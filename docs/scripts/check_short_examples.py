"""Execute the homepage and README snippets and assert their public outputs."""

import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    with tempfile.TemporaryDirectory(prefix="nampy-short-doc-examples-") as directory:
        os.chdir(directory)
        for name in ["index.rst"]:
            namespace = {}
            blocks = re.findall(
                r"^\.\. code-block:: python\n\n((?:(?:   [^\n]*)?\n)+)",
                (ROOT / "docs" / name).read_text(),
                re.M,
            )
            if name == "index.rst":
                blocks = blocks[:1]
            for block in blocks:
                code = "\n".join(
                    line[3:] if line.startswith("   ") else line
                    for line in block.splitlines()
                )
                exec(compile(code, name, "exec"), namespace)
            prediction = np.asarray(namespace["prediction"])
            assert prediction.shape == (len(namespace["data"]),)
            assert np.isfinite(prediction).all()
            namespace["model"].predict_components(
                namespace["data"]
            ).validate_additive_reconstruction()
            plt.close("all")
            print("PASS", name)
        namespace = {}
        for block in re.findall(
            r"```python\n(.*?)\n```", (ROOT / "README.md").read_text(), re.S
        )[:3]:
            exec(compile(block, "README.md", "exec"), namespace)
        parameters = np.asarray(namespace["parameters"])
        assert parameters.shape == (len(namespace["data_dist"]), 2)
        assert np.isfinite(parameters).all() and np.all(parameters[:, 1] > 0)
        namespace["model"].predict_components(
            namespace["data_dist"]
        ).validate_additive_reconstruction()
        print("PASS README statistical examples")


if __name__ == "__main__":
    main()
