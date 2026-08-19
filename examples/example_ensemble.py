#!/usr/bin/env python3
"""
Verification example for EnsembleTreeNAMRegressor on synthetic piecewise-additive data.

Checks
------
1. Predictive quality on a piecewise-constant additive target.
2. Exact additive decomposition:
      output == intercept + sum(returned feature contributions)
3. Recovery of step-like numeric effects.
4. Recovery of categorical feature effects.
5. Comparison against a single TreeNAM baseline.

Run:
    python examples/example_ensemble_treenam.py
"""

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from nampy.models import EnsembleTreeNAMRegressor, TreeNAMRegressor


def main():
    rng = np.random.default_rng(42)
    n = 1600

    # ------------------------------------------------------------------
    # Synthetic mixed data
    # ------------------------------------------------------------------
    X = pd.DataFrame(
        {
            "age": rng.uniform(18, 70, n),
            "income": rng.uniform(0, 120, n),
            "region": rng.choice(["North", "South", "East", "West"], size=n),
        }
    )

    # ------------------------------------------------------------------
    # Piecewise-additive data-generating process
    # ------------------------------------------------------------------
    def f_age(age):
        return np.select(
            [
                age < 30,
                (age >= 30) & (age < 45),
                (age >= 45) & (age < 60),
                age >= 60,
            ],
            [-0.8, -0.1, 0.45, 0.9],
        )

    def f_income(income):
        return np.select(
            [
                income < 25,
                (income >= 25) & (income < 50),
                (income >= 50) & (income < 85),
                income >= 85,
            ],
            [-0.5, 0.0, 0.55, 1.0],
        )

    REGION_EFFECT = {
        "North": -0.35,
        "South": 0.10,
        "East": 0.00,
        "West": 0.25,
    }

    true_age = f_age(X["age"].to_numpy())
    true_income = f_income(X["income"].to_numpy())
    true_region = np.array([REGION_EFFECT[r] for r in X["region"]])

    noise = rng.normal(0.0, 0.18, n)
    y = true_age + true_income + true_region + noise

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    single_model = TreeNAMRegressor(
        tree_depth=6,
        tree_lamda=0.0,
        tree_temperature=0.15,
        use_hard_routing_in_eval=True,
        feature_dropout=0.0,
        numerical_preprocessing="standardization",
        categorical_preprocessing="one_hot",
        cat_cutoff=0.0,
    )

    ensemble_model = EnsembleTreeNAMRegressor(
        num_estimators=5,
        tree_depth=6,
        tree_lamda=0.0,
        tree_temperature=0.15,
        use_hard_routing_in_eval=True,
        feature_dropout=0.0,
        numerical_preprocessing="standardization",
        categorical_preprocessing="one_hot",
        cat_cutoff=0.0,
    )

    with tempfile.TemporaryDirectory() as ckpt_dir_single, tempfile.TemporaryDirectory() as ckpt_dir_ens:
        single_model.fit(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            checkpoint_path=ckpt_dir_single,
            max_epochs=180,
            batch_size=64,
            val_size=0.2,
            patience=20,
            logger=False,
            enable_progress_bar=True,
        )

        ensemble_model.fit(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            checkpoint_path=ckpt_dir_ens,
            max_epochs=180,
            batch_size=64,
            val_size=0.2,
            patience=20,
            logger=False,
            enable_progress_bar=True,
        )

    # ------------------------------------------------------------------
    # Predictions / evaluation
    # ------------------------------------------------------------------
    preds_single = np.asarray(single_model.predict(X_val)).reshape(-1)
    preds_ens = np.asarray(ensemble_model.predict(X_val)).reshape(-1)

    scores_single = single_model.evaluate(
        X_val, y_val, metrics={"MAE": mean_absolute_error, "R2": r2_score}
    )
    scores_ens = ensemble_model.evaluate(
        X_val, y_val, metrics={"MAE": mean_absolute_error, "R2": r2_score}
    )

    print("Single TreeNAM")
    print(
        f"  Validation — MAE: {scores_single['MAE']:.4f}, R2: {scores_single['R2']:.4f}"
    )
    print("Ensemble TreeNAM")
    print(f"  Validation — MAE: {scores_ens['MAE']:.4f}, R2: {scores_ens['R2']:.4f}")

    # ------------------------------------------------------------------
    # Ensemble contributions
    # ------------------------------------------------------------------
    components = ensemble_model.predict_components(X_val)
    feat_vals = dict(components.terms)
    feat_vals["output"] = components.link
    feat_vals["response"] = components.response
    feat_vals["intercept"] = components.intercept
    print("\nReturned contribution keys:")
    print(list(feat_vals.keys()))

    def to_numpy(v):
        t = getattr(v, "detach", lambda: v)()
        arr = np.asarray(t)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        return arr.reshape(-1)

    learned = {k: to_numpy(v) for k, v in feat_vals.items() if k != "output_penalty"}

    # ------------------------------------------------------------------
    # Exact additive decomposition check
    # ------------------------------------------------------------------
    component_keys = [k for k in learned.keys() if k not in {"output", "intercept"}]
    reconstructed = np.zeros_like(preds_ens)

    if "intercept" in learned:
        reconstructed = reconstructed + learned["intercept"]

    for k in component_keys:
        reconstructed = reconstructed + learned[k]

    max_abs_decomp_error = np.max(np.abs(reconstructed - learned["output"]))
    max_abs_pred_error = np.max(np.abs(reconstructed - preds_ens))

    print("\nExact additive decomposition check:")
    print(f"  max |reconstructed - returned output| : {max_abs_decomp_error:.8f}")
    print(f"  max |reconstructed - predict(X_val)|  : {max_abs_pred_error:.8f}")

    # ------------------------------------------------------------------
    # Ground-truth contributions on validation set
    # ------------------------------------------------------------------
    xv = X_val.reset_index(drop=True)
    true_age_val = f_age(xv["age"].to_numpy())
    true_income_val = f_income(xv["income"].to_numpy())
    true_region_val = np.array([REGION_EFFECT[r] for r in xv["region"]])
    true_total_no_noise = true_age_val + true_income_val + true_region_val

    def centered_corr(a, b):
        a = np.asarray(a).reshape(-1)
        b = np.asarray(b).reshape(-1)
        a = a - a.mean()
        b = b - b.mean()
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return np.nan
        return np.corrcoef(a, b)[0, 1]

    print("\nVerification (true DGP vs ensemble learned contributions):")
    print(
        f"  total   true vs predictions correlation: {centered_corr(true_total_no_noise, preds_ens):.4f}"
    )

    if "age" in learned:
        print(
            f"  age     true vs learned correlation:     {centered_corr(true_age_val, learned['age']):.4f}"
        )
    else:
        print("  age     key not found")

    if "income" in learned:
        print(
            f"  income  true vs learned correlation:     {centered_corr(true_income_val, learned['income']):.4f}"
        )
    else:
        print("  income  key not found")

    if "region" in learned:
        print(
            f"  region  true vs learned correlation:     {centered_corr(true_region_val, learned['region']):.4f}"
        )
    else:
        print("  region  key not found")

    # ------------------------------------------------------------------
    # Category-level summaries
    # ------------------------------------------------------------------
    if "region" in learned:
        print("\nMean learned region contribution by category:")
        for region in sorted(REGION_EFFECT.keys()):
            mask = xv["region"].to_numpy() == region
            mean_learned = learned["region"][mask].mean()
            print(
                f"  {region:>5s} | true: {REGION_EFFECT[region]: .3f} | learned mean: {mean_learned: .3f}"
            )

    # ------------------------------------------------------------------
    # Bin-level summaries for the step effects
    # ------------------------------------------------------------------
    if "age" in learned:
        print("\nMean learned age contribution by true step bin:")
        age_bins = [
            ("<30", xv["age"].to_numpy() < 30),
            ("30-45", (xv["age"].to_numpy() >= 30) & (xv["age"].to_numpy() < 45)),
            ("45-60", (xv["age"].to_numpy() >= 45) & (xv["age"].to_numpy() < 60)),
            (">=60", xv["age"].to_numpy() >= 60),
        ]
        for name, mask in age_bins:
            print(
                f"  {name:>5s} | true mean: {true_age_val[mask].mean(): .3f} | "
                f"learned mean: {learned['age'][mask].mean(): .3f}"
            )

    if "income" in learned:
        print("\nMean learned income contribution by true step bin:")
        income_bins = [
            ("<25", xv["income"].to_numpy() < 25),
            ("25-50", (xv["income"].to_numpy() >= 25) & (xv["income"].to_numpy() < 50)),
            ("50-85", (xv["income"].to_numpy() >= 50) & (xv["income"].to_numpy() < 85)),
            (">=85", xv["income"].to_numpy() >= 85),
        ]
        for name, mask in income_bins:
            print(
                f"  {name:>5s} | true mean: {true_income_val[mask].mean(): .3f} | "
                f"learned mean: {learned['income'][mask].mean(): .3f}"
            )

    # ------------------------------------------------------------------
    # Compare single model vs ensemble directly
    # ------------------------------------------------------------------
    print("\nSingle model vs ensemble comparison:")
    print(
        f"  MAE improvement: {scores_single['MAE'] - scores_ens['MAE']:+.4f} (positive means ensemble is better)"
    )
    print(
        f"  R2  improvement: {scores_ens['R2'] - scores_single['R2']:+.4f} (positive means ensemble is better)"
    )

    # ------------------------------------------------------------------
    # Plot: true noiseless signal vs predictions
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    lo = min(true_total_no_noise.min(), preds_single.min(), preds_ens.min())
    hi = max(true_total_no_noise.max(), preds_single.max(), preds_ens.max())

    axes[0].scatter(true_total_no_noise, preds_single, alpha=0.5, s=18)
    axes[0].plot([lo, hi], [lo, hi], "--")
    axes[0].set_xlabel("True noiseless additive signal")
    axes[0].set_ylabel("Single TreeNAM prediction")
    axes[0].set_title("Single TreeNAM")

    axes[1].scatter(true_total_no_noise, preds_ens, alpha=0.5, s=18)
    axes[1].plot([lo, hi], [lo, hi], "--")
    axes[1].set_xlabel("True noiseless additive signal")
    axes[1].set_ylabel("Ensemble TreeNAM prediction")
    axes[1].set_title("Ensemble TreeNAM")

    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "example_ensemble_treenam_plot.png",
    )
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out}")

    print("\nInterpretation:")
    print("  - The additive reconstruction error should be ~0.")
    print("  - Returned keys should include age, income, region, intercept.")
    print("  - Ensemble R2 should often be at least as good as a single TreeNAM.")
    print(
        "  - Learned means within each step bin should roughly match the true step levels."
    )
    print(
        "  - region should be returned as one logical feature block, not region[0], region[1], ..."
    )


if __name__ == "__main__":
    main()
