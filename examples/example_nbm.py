#!/usr/bin/env python3
"""
Verification example for NBMRegressor on synthetic data with:
- known unary effects
- one known pairwise interaction
- train/val split
- full training / evaluation
- explicit recovery checks for main + interaction terms

Run:
    python examples/example_nbm.py
"""

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from nampy.models import NBMRegressor


def main():
    rng = np.random.default_rng(42)
    n = 1200

    # ------------------------------------------------------------------
    # Synthetic data: purely numerical so each feature stays atomic
    # after preprocessing (important for clean NBM verification).
    # ------------------------------------------------------------------
    X = pd.DataFrame(
        {
            "x1": rng.uniform(-2.5, 2.5, n),
            "x2": rng.uniform(-2.0, 2.0, n),
            "x3": rng.uniform(-3.0, 3.0, n),
        }
    )

    # ------------------------------------------------------------------
    # Ground-truth decomposition:
    #   y = f1(x1) + f2(x2) + f3(x3) + f12(x1,x2) + noise
    #
    # We keep the terms smooth and nontrivial so NBM has something real to learn.
    # ------------------------------------------------------------------
    def f1(x1):
        return 0.8 * np.sin(1.5 * x1)

    def f2(x2):
        return 0.5 * (x2**2 - np.mean(x2**2))

    def f3(x3):
        return -0.4 * x3

    def f12(x1, x2):
        return 1.2 * x1 * x2

    true_x1 = f1(X["x1"].to_numpy())
    true_x2 = f2(X["x2"].to_numpy())
    true_x3 = f3(X["x3"].to_numpy())
    true_x1_x2 = f12(X["x1"].to_numpy(), X["x2"].to_numpy())

    noise = rng.normal(0.0, 0.25, n)
    y = true_x1 + true_x2 + true_x3 + true_x1_x2 + noise

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------------------------------------------------------
    # NBM model
    #
    # Key settings for clean verification:
    # - numerical_method="standardization" keeps each feature 1D
    # - nary=[1, 2] enables unary + pairwise terms
    # ------------------------------------------------------------------
    model = NBMRegressor(
        layer_sizes=(128, 64),
        num_bases=64,
        num_subnets=1,
        dropout=0.05,
        bases_dropout=0.05,
        output_penalty=1e-4,
        nary=[1, 2],  # unary + pairwise
        numerical_method="standardization",
        treat_all_integers_as_numerical=True,
    )

    with tempfile.TemporaryDirectory() as ckpt_dir:
        model.fit(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            checkpoint_path=ckpt_dir,
            max_epochs=200,
            batch_size=64,
            val_size=0.2,
            patience=25,
            logger=False,
            enable_progress_bar=True,
        )

    # ------------------------------------------------------------------
    # Predict / evaluate
    # ------------------------------------------------------------------
    preds = model.predict(X_val)
    scores = model.evaluate(
        X_val,
        y_val,
        metrics={"MAE": mean_absolute_error, "R2": r2_score},
    )
    print(f"Validation — MAE: {scores['MAE']:.4f}, R2: {scores['R2']:.4f}")

    # ------------------------------------------------------------------
    # Feature / interaction contributions
    # ------------------------------------------------------------------
    components = model.predict_components(X_val)
    feat_vals = dict(components.terms)
    feat_vals["output"] = components.link
    feat_vals["response"] = components.response
    feat_vals["intercept"] = components.intercept
    print("\nReturned contribution keys:")
    print(list(feat_vals.keys()))

    def to_numpy(v):
        t = getattr(v, "detach", lambda: v)()
        return np.asarray(t)

    learned = {}
    for k, v in feat_vals.items():
        if k in {"output", "intercept", "output_penalty"}:
            continue
        arr = to_numpy(v)

        # For regression, outputs may be shape [N] or [N,1]
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        learned[k] = arr.reshape(-1)

    # ------------------------------------------------------------------
    # Build ground-truth contributions on validation set
    # ------------------------------------------------------------------
    xv = X_val.reset_index(drop=True)

    true_main = {
        "x1": f1(xv["x1"].to_numpy()),
        "x2": f2(xv["x2"].to_numpy()),
        "x3": f3(xv["x3"].to_numpy()),
    }

    true_interactions = {
        "x1:x2": f12(xv["x1"].to_numpy(), xv["x2"].to_numpy()),
    }

    true_total_no_noise = (
        true_main["x1"] + true_main["x2"] + true_main["x3"] + true_interactions["x1:x2"]
    )

    # ------------------------------------------------------------------
    # Correlation helper:
    # center terms first because additive decompositions can differ by constants
    # ------------------------------------------------------------------
    def centered_corr(a, b):
        a = np.asarray(a).reshape(-1)
        b = np.asarray(b).reshape(-1)
        a = a - a.mean()
        b = b - b.mean()
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return np.nan
        return np.corrcoef(a, b)[0, 1]

    print("\nVerification (true DGP vs learned contributions):")
    print(
        f"  total (no noise) vs predictions correlation: {centered_corr(true_total_no_noise, preds):.4f}"
    )

    for key in ["x1", "x2", "x3"]:
        if key in learned:
            print(
                f"  {key:8s} true vs learned correlation: {centered_corr(true_main[key], learned[key]):.4f}"
            )
        else:
            print(f"  {key:8s} not found in learned contributions")

    # Interaction key name can depend on ordering, so check both directions.
    learned_x1_x2 = None
    if "x1:x2" in learned:
        learned_x1_x2 = learned["x1:x2"]
    elif "x2:x1" in learned:
        learned_x1_x2 = learned["x2:x1"]

    if learned_x1_x2 is not None:
        print(
            f"  {'x1:x2':8s} true vs learned correlation: {centered_corr(true_interactions['x1:x2'], learned_x1_x2):.4f}"
        )
    else:
        print("  x1:x2   not found in learned contributions")

    # ------------------------------------------------------------------
    # Check that spurious interactions stay relatively small
    # (these should ideally be much weaker than x1:x2)
    # ------------------------------------------------------------------
    print("\nSpurious interaction magnitudes (smaller is better):")
    for spurious_key in ["x1:x3", "x3:x1", "x2:x3", "x3:x2"]:
        if spurious_key in learned:
            print(
                f"  {spurious_key:8s} mean(|contribution|): {np.mean(np.abs(learned[spurious_key])):.4f}"
            )

    print("\nInterpretation:")
    print("  - R2 should be high (typically > 0.85 on this synthetic task).")
    print(
        "  - Main-effect correlations should be clearly positive and often close to 1."
    )
    print(
        "  - The x1:x2 interaction should also correlate strongly with the true interaction."
    )
    print("  - Spurious interactions should be much smaller than the true x1:x2 term.")

    # ------------------------------------------------------------------
    # Optional plotting through your package API
    # ------------------------------------------------------------------
    try:
        fig = model.plot(X_val, y_val)
        if fig is not None:
            out = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "example_nbm_plot.png",
            )
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"\nPlot saved: {out}")
    except Exception as e:
        print(f"\nPlotting skipped: {e}")


if __name__ == "__main__":
    main()
