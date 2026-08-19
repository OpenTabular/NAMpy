#!/usr/bin/env python3
"""
Single realistic NAM example: NAMRegressor on mixed data with train/val split,
full training, evaluation, and a feature-importance plot.
Run: python examples/example_nam.py
"""

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from nampy.models import NAMRegressor


def main():
    # Realistic-sized synthetic data: numeric + categorical features
    rng = np.random.default_rng(42)
    n = 800
    X = pd.DataFrame(
        {
            "age": rng.uniform(18, 70, n),
            "income": rng.lognormal(10, 0.8, n),
            "region": rng.choice(["North", "South", "East", "West"], n),
            "segment": rng.integers(0, 4, n),
        }
    )
    # Target: additive structure + noise (fixed so we can verify learned effects)
    REGION_EFFECT = {"North": -0.2, "South": 0.1, "East": 0.0, "West": 0.15}
    y = (
        0.02 * X["age"].to_numpy()
        + 0.3
        * (np.log(X["income"].to_numpy()) - np.log(X["income"].to_numpy()).mean())
        / np.log(X["income"].to_numpy()).std()
        + np.array([REGION_EFFECT[r] for r in X["region"]])
        + 0.05 * X["segment"].to_numpy()
        + rng.normal(0, 0.3, n)
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = NAMRegressor(
        layer_sizes=(64, 32),
        dropout=0.05,
        numerical_preprocessing="ple",
        n_bins=50,
        categorical_preprocessing="one_hot",
        cat_cutoff=0.05,
    )

    with tempfile.TemporaryDirectory() as ckpt_dir:
        model.fit(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            checkpoint_path=ckpt_dir,
            max_epochs=150,
            batch_size=64,
            val_size=0.2,
            patience=20,
            logger=False,
            enable_progress_bar=True,
        )

    model.predict(X_val)
    scores = model.evaluate(
        X_val,
        y_val,
        metrics={"MAE": mean_absolute_error, "R2": r2_score},
    )
    print(f"Validation — MAE: {scores['MAE']:.4f}, R2: {scores['R2']:.4f}")

    # Feature contributions (sample)
    components = model.predict_components(X_val)
    feat_vals = dict(components.terms)
    feat_vals["output"] = components.link
    feat_vals["response"] = components.response
    feat_vals["intercept"] = components.intercept
    print(f"Feature / interaction keys: {list(feat_vals.keys())}")

    # --- Verify model learned the true effects (we know the DGP) ---
    def true_contribution_age(x):
        return 0.02 * x["age"].to_numpy()

    def true_contribution_income(x):
        return 1e-5 * np.log(x["income"].to_numpy() + 1)

    def true_contribution_region(x):
        return np.array([REGION_EFFECT[r] for r in x["region"]])

    def true_contribution_segment(x):
        return 0.05 * x["segment"].to_numpy()

    true_age = true_contribution_age(X_val)
    true_income = true_contribution_income(X_val)
    true_region = true_contribution_region(X_val)
    true_segment = true_contribution_segment(X_val)
    true_total_no_noise = true_age + true_income + true_region + true_segment

    def to_numpy(v):
        t = getattr(v, "detach", lambda: v)()
        return np.asarray(t).reshape(-1)

    learned = {k: to_numpy(v) for k, v in feat_vals.items() if k != "output"}
    learned_total = model.predict(X_val)

    def corr(a, b):
        return (
            np.corrcoef(a.ravel(), b.ravel())[0, 1]
            if np.std(a) > 0 and np.std(b) > 0
            else np.nan
        )

    print("\nVerification (true DGP vs learned contributions):")
    print(
        f"  Total (no noise) vs predictions  correlation: {corr(true_total_no_noise, learned_total):.4f}"
    )
    if "age" in learned:
        print(
            f"  age     true vs learned correlation: {corr(true_age, learned['age']):.4f}"
        )
    if "income" in learned:
        print(
            f"  income  true vs learned correlation: {corr(true_income, learned['income']):.4f}"
        )
    if "region" in learned:
        print(
            f"  region  true vs learned correlation: {corr(true_region, learned['region']):.4f}"
        )
    if "segment" in learned:
        print(
            f"  segment true vs learned correlation: {corr(true_segment, learned['segment']):.4f}"
        )
    print("  (Correlation ≈ 1 means the model recovered the effect shape.)")

    # Save plot next to this script
    fig = model.plot(X_val, y_val)
    if fig is not None:
        out = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "example_nam_plot.png"
        )
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved: {out}")


if __name__ == "__main__":
    main()
