"""Generate NAM landing-page outputs from the displayed fitting example."""

import csv
import io
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    torch.set_num_threads(1)
    source = (ROOT / "docs/tutorials/modeling_workflows.rst").read_text()
    section = source.split("Neural Additive Model (NAM)\n", 1)[1]
    block = re.search(
        r"^\.\. code-block:: python\n\n((?:(?:   [^\n]*)?\n)+)", section, re.M
    )[1]
    code = "\n".join(
        line[3:] if line.startswith("   ") else line for line in block.splitlines()
    )
    namespace = {}
    exec(compile(code, "<landing NAM example>", "exec"), namespace)
    prediction = np.asarray(namespace["prediction"]).reshape(-1)
    components = namespace["components"]
    observed = np.asarray(namespace["y"])
    assert np.isfinite(prediction).all()
    frame = pd.DataFrame({"Observed": observed[:5], "Prediction": prediction[:5]})
    frame["Intercept"] = np.broadcast_to(
        components.intercept, np.asarray(components.link).shape
    ).reshape(-1)[:5]
    for name, values in components.terms.items():
        frame[name] = np.asarray(values).reshape(-1)[:5]
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(frame.columns)
    writer.writerows([[f"{value:.4f}" for value in row] for row in frame.to_numpy()])
    output = (
        ".. csv-table:: Predictions and centered components — first five observations\n"
        "   :header-rows: 1\n\n"
        + "".join("   " + line + "\n" for line in stream.getvalue().splitlines())
        + "\nThe intercept and feature contributions sum to the prediction.\n"
    )
    (ROOT / "docs/_generated/home_nam.inc").write_text(output)
    importance = namespace["importance"].sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars = ax.barh(importance["term"], importance["importance"], color="#237f75")
    ax.bar_label(bars, fmt="%.3f", padding=5)
    ax.set(xlabel="Mean absolute centered contribution", title="NAM feature importance")
    ax.set_xlim(0, importance["importance"].max() * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        ROOT / "docs/_static/home_nam_importance.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close("all")
    print(
        "Generated NAM predictions, components, and importance from the landing example."
    )


if __name__ == "__main__":
    main()
