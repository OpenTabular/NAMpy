"""Regenerate the landing page's GAM outputs from its displayed examples."""

import contextlib
import csv
import io
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/_generated"
STATIC = ROOT / "docs/_static"
OUTPUT.mkdir(exist_ok=True)
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)


def table(title, frame):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(frame.columns)
    for row in frame.itertuples(index=False, name=None):
        writer.writerow(
            [f"{v:.4f}" if isinstance(v, (float, np.floating)) else v for v in row]
        )
    return (
        f".. csv-table:: {title}\n   :header-rows: 1\n\n"
        + "".join("   " + line + "\n" for line in stream.getvalue().splitlines())
        + ""
    )


def save(fig, name):
    fig.tight_layout()
    fig.savefig(STATIC / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


source = (ROOT / "docs/_includes/model_showcases.inc").read_text()
blocks = re.findall(
    r"^\.\. code-block:: python\n\n((?:(?:   [^\n]*)?\n)+)", source, re.M
)
namespace = {}
for index, block in enumerate(blocks[:3]):
    code = "\n".join(
        line[3:] if line.startswith("   ") else line for line in block.splitlines()
    )
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code, "<landing example>", "exec"), namespace)
    model, data = namespace["model"], namespace["data"]
    x = data["x"].to_numpy()
    if index == 0:
        components = namespace["components"]
        summary = str(namespace["summary"])
        text = (
            "**Model summary**\n\n.. code-block:: text\n\n"
            + "".join(
                "   " + line.rstrip() + "\n" if line.strip() else "\n"
                for line in summary.splitlines()
            )
            + "\n"
        )
        frame = pd.DataFrame(
            {
                "x": x,
                "group": data["group"],
                "intercept": np.broadcast_to(components.intercept, (len(data),)),
            }
        )
        for name, values in components.terms.items():
            frame[name] = np.asarray(values).reshape(-1)
        frame["prediction"] = namespace["prediction"]
        text += table("Additive components — first five observations", frame.head())
        first = frame.iloc[0]
        addends = [float(first["intercept"])] + [
            float(first[name]) for name in components.terms
        ]
        text += (
            "\nFor the first displayed row, intercept plus fitted contributions gives ``"
            + " + ".join(f"({value:.4f})" for value in addends)
            + f" = {float(first['prediction']):.4f}``, up to displayed rounding. The x and group columns are inputs, not additional summands.\n\n"
        )

        (OUTPUT / "home_gam.inc").write_text(text.rstrip() + "\n")
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        term = next(values for name, values in components.terms.items() if "x" in name)
        axes[0].plot(x, np.asarray(term).reshape(-1), color="#237f75")
        axes[0].set(xlabel="x", ylabel="Contribution", title="Smooth effect of x")
        axes[1].scatter(x, data.y, s=8, alpha=0.3, color="gray", label="Observed")
        for group, color in [("a", "#237f75"), ("b", "#c17c30")]:
            grid = pd.DataFrame({"x": x, "group": group})
            axes[1].plot(x, model.predict(grid), color=color, label=f"Group {group}")
        axes[1].set(xlabel="x", ylabel="y", title="Fitted response")
        axes[1].legend(frameon=False)
        save(fig, "home_gam")
    elif index == 1:
        test = namespace["X_test"]["x"].to_numpy()
        pred = np.asarray(namespace["prediction"]).reshape(-1)
        se = np.asarray(namespace["standard_error"]).reshape(-1)
        frame = pd.DataFrame({"x": test, "prediction": pred, "standard error": se})
        (OUTPUT / "home_gam_adapter.inc").write_text(
            table("Held-out predictions — first five observations", frame.head())
        )
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.scatter(data.x, data.y, s=9, alpha=0.3, color="gray", label="Observed")
        ax.plot(test, pred, color="#237f75", label="Prediction")
        ax.fill_between(
            test,
            pred - 1.96 * se,
            pred + 1.96 * se,
            color="#237f75",
            alpha=0.2,
            label="95% pointwise confidence interval",
        )
        ax.set(xlabel="x", ylabel="y")
        ax.legend(frameon=False)
        save(fig, "home_gam_adapter")
    else:
        data = namespace["data_dist"]
        x = data["x"].to_numpy()
        params = np.asarray(namespace["parameters"])
        raw = np.asarray(namespace["raw_predictors"])
        frame = pd.DataFrame(
            {
                "x": x,
                "mu": params[:, 0],
                "sigma": params[:, 1],
                "eta_mu": raw[:, 0],
                "eta_sigma": raw[:, 1],
            }
        )
        (OUTPUT / "home_gamlss.inc").write_text(
            table(
                "Distribution parameters and link-scale predictors — first five observations",
                frame.head(),
            )
        )
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        axes[0].scatter(x, data.y, s=8, alpha=0.3, color="gray")
        for group, color in [("a", "#237f75"), ("b", "#c17c30")]:
            values = np.asarray(model.predict(pd.DataFrame({"x": x, "group": group})))
            axes[0].plot(x, values[:, 0], color=color, label=f"Group {group}")
        axes[0].set(xlabel="x", ylabel="mu", title="Conditional mean")
        axes[0].legend(frameon=False)
        axes[1].plot(x, params[:, 1], color="#237f75")
        axes[1].set(xlabel="x", ylabel="sigma", title="Conditional standard deviation")
        save(fig, "home_gamlss")
print("Generated three plots and example output tables.")
