"""Generate teaching figures and tables from the same complete example runs."""

import contextlib
import hashlib
import io
import json
import logging
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "docs/_static"
OUTPUT = ROOT / "docs/_generated"


def csv_table(name, title, frame):
    stream = io.StringIO()
    frame.to_csv(stream, index=False, float_format="%.4f")
    text = f".. csv-table:: {title}\n   :header-rows: 1\n\n"
    text += "".join("   " + line + "\n" for line in stream.getvalue().splitlines())
    (OUTPUT / (name + ".inc")).write_text(text.rstrip() + "\n")


def feature_component(parts, feature):
    names = [
        name
        for name, meta in parts.term_metadata.items()
        if meta.features == (feature,)
    ]
    if not names:
        names = [
            name
            for name in parts.terms
            if name == feature or name.startswith(feature + "[")
        ]
    if not names:
        raise AssertionError(f"No contribution for {feature}")
    return sum(np.asarray(parts.terms[name]) for name in names)


def save(fig, name):
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(
        STATIC / (name + ".png"), dpi=150, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        tree = runpy.run_path(str(ROOT / "examples/example_ensemble.py"))["main"]()
        quantile = runpy.run_path(str(ROOT / "examples/example_qnam.py"))["main"]()
    # Existing prediction figures remain, with their intended comparison stated.
    import shutil

    shutil.copyfile(
        ROOT / "examples/example_ensemble_treenam_plot.png",
        STATIC / "detailed_tree_effects.png",
    )
    shutil.copyfile(
        ROOT / "examples/example_qnam_plot.png", STATIC / "detailed_quantile_median.png"
    )
    X_train, X_test = tree["X_train"], tree["X_test"]
    metrics = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    regions = sorted(tree["region_effect"])
    for name, model in [
        ("Single tree", tree["single"]),
        ("Joint ensemble", tree["ensemble"]),
    ]:
        prediction = model.predict(X_test)
        metrics.append(
            {
                "Model": name,
                "Test n": len(X_test),
                "Noisy outcome MAE": mean_absolute_error(tree["y_test"], prediction),
                "Noisy outcome R2": r2_score(tree["y_test"], prediction),
                "Noiseless signal MAE": mean_absolute_error(
                    tree["noiseless_test"], prediction
                ),
            }
        )
        for axis, feature in zip(axes[:2], ["age", "income"], strict=True):
            values = np.linspace(X_train[feature].min(), X_train[feature].max(), 240)
            grid = pd.DataFrame(
                {
                    "age": np.full(240, X_train.age.median()),
                    "income": np.full(240, X_train.income.median()),
                    "region": regions[0],
                }
            )
            grid[feature] = values
            parts = model.predict_components(grid, center=True, reference_X=X_train)
            learned = feature_component(parts, feature).reshape(-1)
            axis.plot(values, learned, label=name)
            function = tree["f_" + feature]
            if name == "Single tree":
                axis.plot(
                    values,
                    function(values) - function(X_train[feature].to_numpy()).mean(),
                    "k--",
                    label="Centered truth",
                )
            axis.set(
                title=feature, xlabel=feature, ylabel="Centered response contribution"
            )
        grid = pd.DataFrame(
            {
                "age": X_train.age.median(),
                "income": X_train.income.median(),
                "region": regions,
            }
        )
        parts = model.predict_components(grid, center=True, reference_X=X_train)
        axes[2].plot(
            regions, feature_component(parts, "region").reshape(-1), "o-", label=name
        )
    truth_reference = np.mean([tree["region_effect"][r] for r in X_train.region])
    axes[2].plot(
        regions,
        [tree["region_effect"][r] - truth_reference for r in regions],
        "k--",
        label="Centered truth",
    )
    axes[2].set(title="region", ylabel="Centered response contribution")
    for axis in axes:
        axis.legend(fontsize=8)
    tree_panels = [axis.get_title() for axis in fig.axes]
    tree_lines = [len(axis.lines) for axis in fig.axes]
    save(fig, "detailed_tree_contributions")
    csv_table(
        "detailed_tree_metrics",
        "Untouched test outcomes and noiseless signal — lower MAE, higher R²",
        pd.DataFrame(metrics),
    )

    levels = np.asarray(quantile["levels"])
    prediction, y = quantile["prediction"], np.asarray(quantile["y_test"])
    observed = np.mean(y[:, None] <= prediction, axis=0)
    coverage = pd.DataFrame(
        {
            "Quantile level": levels,
            "Observed test fraction below": observed,
            "Test n": len(y),
            "MAE versus known quantile": np.mean(
                np.abs(prediction - quantile["true_quantiles"]), axis=0
            ),
        }
    )
    csv_table(
        "detailed_quantile_coverage",
        "Quantile calibration on untouched test rows",
        coverage,
    )
    interval = pd.DataFrame(
        [
            {
                "Target coverage": 0.8,
                "Observed test coverage": float(
                    np.mean((y >= prediction[:, 0]) & (y <= prediction[:, -1]))
                ),
                "Mean width": float(np.mean(prediction[:, -1] - prediction[:, 0])),
                "Test n": len(y),
                "Pinball loss (sum levels, mean rows)": quantile["model"].evaluate(
                    quantile["X_test"], y, metrics={}
                )["pinball_loss"],
            }
        ]
    )
    csv_table("detailed_quantile_interval", "80% interval width and coverage", interval)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(levels, observed, "o-", label=f"Test n={len(y)}")
    axes[0].plot([0, 1], [0, 1], "k--", label="Ideal calibration")
    axes[0].set(
        xlabel="Nominal quantile level",
        ylabel="Observed fraction below",
        title="Test quantile calibration",
    )
    model, reference = quantile["model"], quantile["X_train"]
    values = np.linspace(reference.age.min(), reference.age.max(), 180)
    grid = pd.DataFrame(
        {
            "age": values,
            "income": reference.income.median(),
            "region": reference.region.iloc[0],
        }
    )
    parts = model.predict_components(grid, center=True, reference_X=reference)
    contribution = feature_component(parts, "age")
    for i, level in enumerate(levels):
        line = axes[1].plot(values, contribution[:, i], label=f"Fitted q={level:g}")[0]

        def function(a, z=quantile["q_z"][i]):
            return quantile["age_function"](a) + z * quantile["age_scale"](a)

        axes[1].plot(
            values,
            function(values) - function(reference.age.to_numpy()).mean(),
            "--",
            color=line.get_color(),
            label=f"Truth q={level:g}",
        )
    axes[1].set(
        xlabel="Age",
        ylabel="Centered quantile contribution",
        title="Known and fitted age effects",
    )
    for axis in axes:
        axis.legend(fontsize=8)
    quantile_panels = [axis.get_title() for axis in fig.axes]
    quantile_lines = [len(axis.lines) for axis in fig.axes]
    save(fig, "detailed_quantile_calibration")
    manifest = {
        "tree": {"panels": tree_panels, "lines": tree_lines, "test_rows": len(X_test)},
        "quantile": {
            "panels": quantile_panels,
            "lines": quantile_lines,
            "test_rows": len(y),
            "levels": levels.tolist(),
        },
        "source_sha256": {
            name: hashlib.sha256((ROOT / "examples" / name).read_bytes()).hexdigest()
            for name in ["example_ensemble.py", "example_qnam.py"]
        },
        "centering": "empirical training rows",
        "evaluation": "test rows excluded from early stopping",
    }
    (OUTPUT / "detailed_assets.json").write_text(json.dumps(manifest, indent=2))
    print(
        "PASS: tree and quantile plots, tables, and manifest generated from the same runs"
    )


if __name__ == "__main__":
    main()
