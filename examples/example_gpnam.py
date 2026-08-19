#!/usr/bin/env python3
"""
Verification example for GPNAMRegressor on mixed additive data.

Checks:
1. Predictive quality on a known smooth additive signal.
2. Exact additive decomposition:
      output == intercept + sum(returned feature contributions)
3. Recovery of smooth numeric effects.
4. Recovery of one-hot categorical effects via grouped region[...] terms.

Run:
    python examples/example_gpnam.py
"""

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from nampy.models import GPNAMRegressor


def main():
    rng = np.random.default_rng(42)
    n = 1400

    # ------------------------------------------------------------------
    # Synthetic mixed data
    # ------------------------------------------------------------------
    X = pd.DataFrame(
        {
            "age": rng.uniform(18, 70, n),
            "income": rng.lognormal(mean=10.0, sigma=0.6, size=n),
            "region": rng.choice(["North", "South", "East", "West"], size=n),
        }
    )

    # ------------------------------------------------------------------
    # Smooth additive data-generating process
    # GP-NAM should be a good fit for this kind of signal.
    # ------------------------------------------------------------------
    log_income = np.log(X["income"].to_numpy())
    income_z = (log_income - log_income.mean()) / log_income.std()

    def f_age(age):
        # smooth nonlinear effect
        x = (age - 18.0) / (70.0 - 18.0)
        return 0.9 * np.sin(2.5 * np.pi * x)

    def f_income(z):
        # smooth nonlinear effect on standardized log-income
        return 0.55 * z - 0.18 * (z**2)

    REGION_EFFECT = {
        "North": -0.30,
        "South": 0.10,
        "East": 0.00,
        "West": 0.22,
    }

    true_age = f_age(X["age"].to_numpy())
    true_income = f_income(income_z)
    true_region = np.array([REGION_EFFECT[r] for r in X["region"]])

    noise = rng.normal(0.0, 0.20, n)
    y = true_age + true_income + true_region + noise

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------------------------------------------------------
    # GP-NAM model
    #
    # Important:
    # - standardize numerics so one global kernel_width is sensible
    # - one-hot categoricals so returned keys include region[0], ...
    # ------------------------------------------------------------------
    model = GPNAMRegressor(
        kernel_width=0.8,
        rff_num_feat=128,
        numerical_preprocessing="standardization",
        categorical_preprocessing="one_hot",
        cat_cutoff=0.0,
    )

    with tempfile.TemporaryDirectory() as ckpt_dir:
        model.fit(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            checkpoint_path=ckpt_dir,
            max_epochs=180,
            batch_size=64,
            val_size=0.2,
            patience=20,
            logger=False,
            enable_progress_bar=True,
        )

    preds = np.asarray(model.predict(X_val)).reshape(-1)
    scores = model.evaluate(
        X_val,
        y_val,
        metrics={"MAE": mean_absolute_error, "R2": r2_score},
    )
    print(f"Validation — MAE: {scores['MAE']:.4f}, R2: {scores['R2']:.4f}")

    # ------------------------------------------------------------------
    # Returned additive contributions
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
        arr = np.asarray(t)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        return arr.reshape(-1)

    learned = {k: to_numpy(v) for k, v in feat_vals.items()}

    # ------------------------------------------------------------------
    # Build ground-truth contributions on validation set
    # ------------------------------------------------------------------
    xv = X_val.reset_index(drop=True)
    log_income_val = np.log(xv["income"].to_numpy())
    income_z_val = (log_income_val - log_income.mean()) / log_income.std()

    true_age_val = f_age(xv["age"].to_numpy())
    true_income_val = f_income(income_z_val)
    true_region_val = np.array([REGION_EFFECT[r] for r in xv["region"]])
    true_total_no_noise = true_age_val + true_income_val + true_region_val

    # ------------------------------------------------------------------
    # Group one-hot region terms back to the logical feature effect
    # ------------------------------------------------------------------
    region_keys = sorted([k for k in learned if k.startswith("region[")])

    print("\nAtomic one-hot categorical keys:")
    print(f"  region keys: {region_keys}")
    if not region_keys:
        raise RuntimeError(
            "No region[...] keys were returned. One-hot categorical handling may be broken."
        )

    learned_region_total = sum(learned[k] for k in region_keys)

    # ------------------------------------------------------------------
    # Check exact additive decomposition
    # ------------------------------------------------------------------
    component_keys = [k for k in learned.keys() if k not in {"output", "intercept"}]
    reconstructed = np.zeros_like(preds)

    if "intercept" in learned:
        reconstructed = reconstructed + learned["intercept"]

    for k in component_keys:
        reconstructed = reconstructed + learned[k]

    max_abs_decomp_error = np.max(np.abs(reconstructed - learned["output"]))
    max_abs_pred_error = np.max(np.abs(reconstructed - preds))

    print("\nExact additive decomposition check:")
    print(f"  max |reconstructed - returned output| : {max_abs_decomp_error:.8f}")
    print(f"  max |reconstructed - predict(X_val)|  : {max_abs_pred_error:.8f}")

    # ------------------------------------------------------------------
    # Correlation helper
    # Center before correlating because additive decompositions can shift by constants
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
        f"  total   true vs predictions correlation: {centered_corr(true_total_no_noise, preds):.4f}"
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

    print(
        f"  region  true vs learned correlation:     {centered_corr(true_region_val, learned_region_total):.4f}"
    )

    # ------------------------------------------------------------------
    # Category-level region summaries
    # ------------------------------------------------------------------
    print("\nMean learned region contribution by category:")
    for region in sorted(REGION_EFFECT.keys()):
        mask = xv["region"].to_numpy() == region
        mean_learned = learned_region_total[mask].mean()
        print(
            f"  {region:>5s} | true: {REGION_EFFECT[region]: .3f} | learned mean: {mean_learned: .3f}"
        )

    # ------------------------------------------------------------------
    # Inspect atomic region terms
    # ------------------------------------------------------------------
    print("\nMean absolute contribution of each one-hot atomic region term:")
    for k in region_keys:
        print(f"  {k:10s}: {np.mean(np.abs(learned[k])):.4f}")

    # ------------------------------------------------------------------
    # Optional simple plot: predicted vs true noiseless signal
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(true_total_no_noise, preds, alpha=0.5, s=18)
    lo = min(true_total_no_noise.min(), preds.min())
    hi = max(true_total_no_noise.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "--")
    ax.set_xlabel("True noiseless additive signal")
    ax.set_ylabel("GP-NAM prediction")
    ax.set_title("GP-NAM verification")

    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "example_gpnam_plot.png",
    )
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out}")

    print("\nInterpretation:")
    print("  - R2 should be high on this smooth additive task.")
    print("  - The additive reconstruction error should be ~0.")
    print("  - age / income correlations should be clearly positive, often high.")
    print("  - Summed region[...] terms should track the true region effect well.")
    print(
        "  - Seeing region[0], region[1], ... is expected because GP-NAM is atomic over scalar post-preprocessing columns."
    )


if __name__ == "__main__":
    main()
