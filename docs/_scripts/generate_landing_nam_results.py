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

from nampy.plotting import prepared_from_contributions
from nampy.plotting.effects import add_density_shading

ROOT = Path(__file__).resolve().parents[2]


def _nam_panels(namespace):
    """Six equal-size panels from one fitted NAM and its declared experiment."""
    X, components = namespace["X"], namespace["components"]
    prepared = prepared_from_contributions(X, components.terms)
    importance = namespace["importance"].set_index("term")
    names = {
        "median_income": "Median income",
        "house_age": "House age",
        "latitude": "Latitude",
        "rooms_per_household": "Rooms / household",
    }
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7))
    for axis, values, feature in zip(
        axes.flat, prepared["pd"], X.columns, strict=False
    ):
        x = values["x"]
        add_density_shading(axis, values["raw"])
        axis.plot(x, values["fit"], color="#237f75", label="Fitted NAM")
        truth = namespace["truth_functions"][feature]
        axis.plot(
            x, truth(x) - truth(X[feature].to_numpy()).mean(),
            "--", color="#c17c30", label="Known centered truth",
        )
        axis.set(
            xlabel=names[feature], ylabel="Centered contribution",
            title=names[feature],
        )
        axis.legend(fontsize=8, frameon=False)
    labels = [names[name] for name in X.columns]
    effect = [float(importance.loc[name, "importance"]) for name in X.columns]
    information = [namespace["fisher_per_parameter"][name] for name in X.columns]
    assert np.isfinite(information).all() and np.all(np.asarray(information) >= 0)
    for axis, values, title, xlabel in (
        (axes.flat[4], effect, "Contribution magnitude", "Mean absolute centered contribution"),
        (axes.flat[5], information, "Mean Fisher information", "Per network parameter (known σ = 0.3)"),
    ):
        bars = axis.barh(labels, values, color="#237f75")
        axis.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
        axis.set(xlabel=xlabel, title=title, xlim=(0, max(values) * 1.24))
        axis.invert_yaxis()
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(
        ROOT / "docs/_static/home_nam.png", dpi=180,
        bbox_inches="tight", facecolor="white",
    )
    print("NAM mean diagonal Fisher by feature network:", dict(zip(X.columns, information, strict=True)))
    plt.close(figure)


def main():
    torch.set_num_threads(1)
    source = (ROOT / "docs/_includes/model_showcases.inc").read_text()
    section = source.split("Neural Additive Model (NAM)\n", 1)[1]
    block = re.search(
        r"^\.\. code-block:: python\n\n((?:(?:   [^\n]*)?\n)+)", section, re.M
    )[1]
    code = "\n".join(
        line[3:] if line.startswith("   ") else line for line in block.splitlines()
    )
    namespace = {}
    exec(compile(code, "<landing NAM example>", "exec"), namespace)
    namespace["figures"][0].savefig(
        ROOT / "docs/_static/homepage_demo_light.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
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
    _nam_panels(namespace)
    # The distributional panel is generated from its displayed block as well.
    section = source.split("Neural distributional regression\n", 1)[1]
    block = re.search(
        r"^\.\. code-block:: python\n\n((?:(?:   [^\n]*)?\n)+)", section, re.M
    )[1]
    code = "\n".join(
        line[3:] if line.startswith("   ") else line for line in block.splitlines()
    )
    exec(compile(code, "<distributional showcase>", "exec"), namespace)
    namespace["figures"][0].savefig(
        ROOT / "docs/_static/readme_distributional_effects.png",
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
