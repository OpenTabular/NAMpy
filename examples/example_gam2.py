#!/usr/bin/env python3
"""
Verification example for Gaussian GAM smoothing selection.

Checks:
1. Fixed smoothing works.
2. GCV smoothing selection works.
3. REML smoothing selection works.
4. Additive decomposition is exact.
5. Learned smooth effects correlate with the true DGP.

Run:
    python examples/example_gam_smoothing.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from nampy.models import GAMRegressor


def corr(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def summarize_model(name, model, X_val, y_val, true_x1, true_x2, true_total):
    preds = model.predict(X_val)
    scores = model.evaluate(
        X_val,
        y_val,
        metrics={
            "MAE": mean_absolute_error,
            "R2": r2_score,
        },
    )

    feat_vals = model.predict_feature_vals(X_val)

    def to_numpy(v):
        arr = np.asarray(v)
        if arr.ndim == 0:
            return float(arr)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        return arr.reshape(-1)

    learned = {k: to_numpy(v) for k, v in feat_vals.items()}

    reconstructed = np.zeros_like(preds, dtype=float)
    if "intercept" in learned:
        reconstructed += float(learned["intercept"])
    if "x1" in learned:
        reconstructed += learned["x1"]
    if "x2" in learned:
        reconstructed += learned["x2"]

    recon_err_1 = np.max(np.abs(reconstructed - learned["output"]))
    recon_err_2 = np.max(np.abs(reconstructed - preds))

    print(f"\n{name}")
    print(f"  Validation — MAE: {scores['MAE']:.4f}, R2: {scores['R2']:.4f}")
    fit_result = model.gam_.fit_result()
    print(f"  smoothing params: {np.round(fit_result.smoothing_params, 6)}")
    print(f"  EDF total: {fit_result.edf_total:.4f}")
    print(f"  EDF by term: {np.round(fit_result.edf_by_term, 4)}")

    print("  Exact additive decomposition check:")
    print(f"    max |reconstructed - returned output| : {recon_err_1:.8f}")
    print(f"    max |reconstructed - predict(X_val)|  : {recon_err_2:.8f}")

    print("  Verification (true DGP vs learned contributions):")
    print(f"    total true vs learned correlation: {corr(true_total, preds):.4f}")
    if "x1" in learned:
        print(
            f"    x1    true vs learned correlation: {corr(true_x1, learned['x1']):.4f}"
        )
    if "x2" in learned:
        print(
            f"    x2    true vs learned correlation: {corr(true_x2, learned['x2']):.4f}"
        )

    return preds, learned, scores


def main():
    rng = np.random.default_rng(42)
    n = 1200

    # ------------------------------------------------------------------
    # Smooth additive Gaussian DGP
    # ------------------------------------------------------------------
    X = pd.DataFrame(
        {
            "x1": rng.uniform(-2.5, 2.5, n),
            "x2": rng.uniform(-2.0, 2.0, n),
        }
    )

    x1 = X["x1"].to_numpy()
    x2 = X["x2"].to_numpy()

    def f1(x):
        return 1.2 * np.sin(1.8 * x)

    def f2(x):
        return 0.8 * (x**2) - 0.9 * x

    intercept_true = 0.3
    noise = rng.normal(0, 0.35, n)

    y_true_no_noise = intercept_true + f1(x1) + f2(x2)
    y = y_true_no_noise + noise

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    xv = X_val.reset_index(drop=True)
    true_x1 = f1(xv["x1"].to_numpy())
    true_x2 = f2(xv["x2"].to_numpy())
    true_total = intercept_true + true_x1 + true_x2

    # ------------------------------------------------------------------
    # 1) Fixed smoothing
    # ------------------------------------------------------------------
    model_fixed = GAMRegressor(
        family="gaussian",
        k=12,
        smoothing_params=[1.0, 1.0],
        fit_intercept=True,
        optimize_smoothing=False,
        smoothing_method="fixed",
    )
    model_fixed.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 2) GCV smoothing selection
    # ------------------------------------------------------------------
    model_gcv = GAMRegressor(
        family="gaussian",
        k=12,
        smoothing_params=[1.0, 1.0],
        fit_intercept=True,
        optimize_smoothing=True,
        smoothing_method="gcv",
    )
    model_gcv.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 3) REML smoothing selection
    # ------------------------------------------------------------------
    model_reml = GAMRegressor(
        family="gaussian",
        k=12,
        smoothing_params=[1.0, 1.0],
        fit_intercept=True,
        optimize_smoothing=True,
        smoothing_method="reml",
    )
    model_reml.fit(X_train, y_train)

    preds_fixed, learned_fixed, scores_fixed = summarize_model(
        "Fixed smoothing",
        model_fixed,
        X_val,
        y_val,
        true_x1,
        true_x2,
        true_total,
    )

    preds_gcv, learned_gcv, scores_gcv = summarize_model(
        "GCV smoothing",
        model_gcv,
        X_val,
        y_val,
        true_x1,
        true_x2,
        true_total,
    )

    preds_reml, learned_reml, scores_reml = summarize_model(
        "REML smoothing",
        model_reml,
        X_val,
        y_val,
        true_x1,
        true_x2,
        true_total,
    )

    # ------------------------------------------------------------------
    # Compare methods
    # ------------------------------------------------------------------
    print("\nComparison:")
    print(f"  Fixed R2 : {scores_fixed['R2']:.4f}")
    print(f"  GCV R2   : {scores_gcv['R2']:.4f}")
    print(f"  REML R2  : {scores_reml['R2']:.4f}")
    print(
        f"  GCV - Fixed  R2 improvement : {scores_gcv['R2'] - scores_fixed['R2']:+.4f}"
    )
    print(
        f"  REML - Fixed R2 improvement : {scores_reml['R2'] - scores_fixed['R2']:+.4f}"
    )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    axes = axes.ravel()

    x1_val = xv["x1"].to_numpy()
    x2_val = xv["x2"].to_numpy()

    order1 = np.argsort(x1_val)
    order2 = np.argsort(x2_val)

    # Fixed
    axes[0].plot(x1_val[order1], true_x1[order1], label="true")
    axes[0].plot(x1_val[order1], learned_fixed["x1"][order1], label="fixed")
    axes[0].set_title("x1 effect — fixed")
    axes[0].legend()

    axes[1].plot(x2_val[order2], true_x2[order2], label="true")
    axes[1].plot(x2_val[order2], learned_fixed["x2"][order2], label="fixed")
    axes[1].set_title("x2 effect — fixed")
    axes[1].legend()

    # GCV
    axes[2].plot(x1_val[order1], true_x1[order1], label="true")
    axes[2].plot(x1_val[order1], learned_gcv["x1"][order1], label="gcv")
    axes[2].set_title("x1 effect — GCV")
    axes[2].legend()

    axes[3].plot(x2_val[order2], true_x2[order2], label="true")
    axes[3].plot(x2_val[order2], learned_gcv["x2"][order2], label="gcv")
    axes[3].set_title("x2 effect — GCV")
    axes[3].legend()

    # REML
    axes[4].plot(x1_val[order1], true_x1[order1], label="true")
    axes[4].plot(x1_val[order1], learned_reml["x1"][order1], label="reml")
    axes[4].set_title("x1 effect — REML")
    axes[4].legend()

    axes[5].plot(x2_val[order2], true_x2[order2], label="true")
    axes[5].plot(x2_val[order2], learned_reml["x2"][order2], label="reml")
    axes[5].set_title("x2 effect — REML")
    axes[5].legend()

    fig.tight_layout()

    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "example_gam_smoothing_plot.png",
    )
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out}")

    print("\nInterpretation:")
    print("  - Additive reconstruction error should be ~0 for all models.")
    print("  - GCV and REML should produce finite positive smoothing parameters.")
    print("  - EDFs should be sensible (not exploding, not all collapsed).")
    print("  - Learned x1/x2 effects should correlate strongly with the true DGP.")
    print(
        "  - GCV / REML often match or improve on an arbitrary fixed smoothing choice."
    )


if __name__ == "__main__":
    main()
