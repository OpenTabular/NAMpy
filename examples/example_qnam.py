#!/usr/bin/env python3
"""
Verification example for QNAM on synthetic additive quantile data.

What this checks
----------------
1. Predicted quantiles match known true conditional quantiles.
2. Predicted quantiles are non-crossing.
3. Returned feature contributions are non-crossing.
4. One-hot categorical contributions (region[...]) can be summed back to the
   logical categorical effect and compared to truth.

Run:
    python examples/example_qnam.py
"""

import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from nampy.models import QNAM


def main():
    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # Quantiles to fit / verify
    # ------------------------------------------------------------------
    quantiles = [0.1, 0.5, 0.9]
    q_tensor = torch.tensor(quantiles, dtype=torch.float32)
    q_z = torch.distributions.Normal(0.0, 1.0).icdf(q_tensor).numpy()

    # ------------------------------------------------------------------
    # Synthetic mixed data: numeric + categorical
    # ------------------------------------------------------------------
    n = 1400
    X = pd.DataFrame({
        "age": rng.uniform(18, 70, n),
        "income": rng.lognormal(mean=10.0, sigma=0.55, size=n),
        "region": rng.choice(["North", "South", "East", "West"], size=n),
    })

    log_income = np.log(X["income"].to_numpy())
    income_z = (log_income - log_income.mean()) / log_income.std()

    # ------------------------------------------------------------------
    # Data-generating process
    #
    # Y | X = mu(X) + sigma(X) * eps,  eps ~ N(0,1)
    #
    # We make both mu(X) and sigma(X) additive:
    #   mu(X)    = intercept_mu + mu_age + mu_income + mu_region
    #   sigma(X) = intercept_sigma + sigma_age + sigma_income + sigma_region
    #
    # Then each conditional quantile is also additive:
    #   Q_q(Y|X) = mu(X) + z_q * sigma(X)
    # where z_q = Phi^{-1}(q).
    # ------------------------------------------------------------------
    INTERCEPT_MU = 0.20
    INTERCEPT_SIGMA = 0.18

    REGION_MU = {
        "North": -0.25,
        "South": 0.12,
        "East": 0.00,
        "West": 0.20,
    }

    REGION_SIGMA = {
        "North": 0.03,
        "South": 0.05,
        "East": 0.02,
        "West": 0.06,
    }

    def mu_age(age):
        return 0.025 * (age - 40.0)

    def sigma_age(age):
        # Always positive
        return 0.08 * (np.sin(age / 8.0) + 1.6)

    def mu_income(z):
        return 0.42 * z

    def sigma_income(z):
        # Always positive
        return 0.07 * (np.abs(z) + 0.7)

    mu_age_all = mu_age(X["age"].to_numpy())
    sigma_age_all = sigma_age(X["age"].to_numpy())
    mu_income_all = mu_income(income_z)
    sigma_income_all = sigma_income(income_z)
    mu_region_all = np.array([REGION_MU[r] for r in X["region"]])
    sigma_region_all = np.array([REGION_SIGMA[r] for r in X["region"]])

    mu_all = INTERCEPT_MU + mu_age_all + mu_income_all + mu_region_all
    sigma_all = (
        INTERCEPT_SIGMA
        + sigma_age_all
        + sigma_income_all
        + sigma_region_all
    )

    eps = rng.normal(0.0, 1.0, n)
    y = mu_all + sigma_all * eps

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------------------------------------------------------
    # Fit QNAM
    # ------------------------------------------------------------------
    model = QNAM(
        layer_sizes=(128, 64),
        dropout=0.05,
        feature_dropout=0.0,
        numerical_preprocessing="standardization",
        categorical_preprocessing="one_hot",
        cat_cutoff=0.0,
        monotone_transform="softplus",
        min_increment=0.0,
    )

    with tempfile.TemporaryDirectory() as ckpt_dir:
        model.fit(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            distributional_kwargs={"quantiles": quantiles},
            checkpoint_path=ckpt_dir,
            max_epochs=180,
            batch_size=64,
            val_size=0.2,
            patience=20,
            logger=False,
            enable_progress_bar=True,
        )

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    preds = np.asarray(model.predict(X_val))
    if preds.ndim != 2 or preds.shape[1] != len(quantiles):
        raise RuntimeError(
            f"Expected predictions of shape [N, {len(quantiles)}], got {preds.shape}"
        )

    xv = X_val.reset_index(drop=True)
    log_income_val = np.log(xv["income"].to_numpy())
    income_z_val = (log_income_val - log_income.mean()) / log_income.std()

    mu_age_val = mu_age(xv["age"].to_numpy())
    sigma_age_val = sigma_age(xv["age"].to_numpy())
    mu_income_val = mu_income(income_z_val)
    sigma_income_val = sigma_income(income_z_val)
    mu_region_val = np.array([REGION_MU[r] for r in xv["region"]])
    sigma_region_val = np.array([REGION_SIGMA[r] for r in xv["region"]])

    mu_val = INTERCEPT_MU + mu_age_val + mu_income_val + mu_region_val
    sigma_val = (
        INTERCEPT_SIGMA
        + sigma_age_val
        + sigma_income_val
        + sigma_region_val
    )

    true_quantiles = np.column_stack([mu_val + z * sigma_val for z in q_z])

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def centered_corr(a, b):
        a = np.asarray(a).reshape(-1)
        b = np.asarray(b).reshape(-1)
        a = a - a.mean()
        b = b - b.mean()
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return np.nan
        return np.corrcoef(a, b)[0, 1]

    def to_numpy(v):
        t = getattr(v, "detach", lambda: v)()
        return np.asarray(t)

    # ------------------------------------------------------------------
    # Quantile-level verification
    # ------------------------------------------------------------------
    print("\nPrediction quality by quantile:")
    for i, q in enumerate(quantiles):
        mae = np.mean(np.abs(preds[:, i] - true_quantiles[:, i]))
        corr = centered_corr(preds[:, i], true_quantiles[:, i])
        empirical_coverage = np.mean(np.asarray(y_val) <= preds[:, i])
        print(
            f"  q={q:.2f} | MAE vs true q: {mae:.4f} | "
            f"corr(pred, true q): {corr:.4f} | "
            f"empirical coverage: {empirical_coverage:.4f}"
        )

    noncross_rate = np.mean(np.all(np.diff(preds, axis=1) >= -1e-8, axis=1))
    min_gap = np.min(np.diff(preds, axis=1))
    print(f"\nNon-crossing prediction rate: {noncross_rate:.4f}")
    print(f"Minimum adjacent quantile gap in predictions: {min_gap:.6f}")

    # ------------------------------------------------------------------
    # Feature contributions
    # ------------------------------------------------------------------
    feat_vals = model.predict_feature_vals(X_val)
    print("\nReturned contribution keys:")
    print(list(feat_vals.keys()))

    learned = {}
    for k, v in feat_vals.items():
        arr = to_numpy(v)
        if arr.ndim == 1:
            learned[k] = arr
        elif arr.ndim == 2:
            learned[k] = arr
        else:
            raise RuntimeError(f"Unexpected shape for key {k}: {arr.shape}")

    # Verify returned contributions are also non-crossing
    print("\nContribution monotonicity check:")
    for k, arr in learned.items():
        if k == "output":
            continue
        if arr.ndim == 1:
            diffs = np.diff(arr)
        else:
            diffs = np.diff(arr, axis=1)
        worst_gap = np.min(diffs) if diffs.size > 0 else np.nan
        print(f"  {k:12s} min adjacent gap: {worst_gap:.6f}")

    # ------------------------------------------------------------------
    # Compare learned feature contributions to true quantile contributions
    # ------------------------------------------------------------------
    true_age_q = np.column_stack([mu_age_val + z * sigma_age_val for z in q_z])
    true_income_q = np.column_stack([mu_income_val + z * sigma_income_val for z in q_z])
    true_region_q = np.column_stack([mu_region_val + z * sigma_region_val for z in q_z])
    true_intercept_q = INTERCEPT_MU + q_z * INTERCEPT_SIGMA

    # Region contributions: model may return one key "region" [n, n_quantiles] or
    # one-hot keys "region[North]", "region[South]", etc. Handle both.
    region_keys = sorted([k for k in learned if k.startswith("region[")])
    if region_keys:
        learned_region_q = sum(learned[k] for k in region_keys)
    elif "region" in learned:
        learned_region_q = learned["region"]
    else:
        learned_region_q = None

    print("\nRecovered contributions (true vs learned, centered correlation):")
    for i, q in enumerate(quantiles):
        print(f"  q={q:.2f}")
        if "age" in learned:
            print(
                f"    age       corr: {centered_corr(true_age_q[:, i], learned['age'][:, i]):.4f}"
            )
        if "income" in learned:
            print(
                f"    income    corr: {centered_corr(true_income_q[:, i], learned['income'][:, i]):.4f}"
            )
        if learned_region_q is not None:
            print(
                f"    region    corr: {centered_corr(true_region_q[:, i], learned_region_q[:, i]):.4f}"
            )
        print(
            f"    output    corr: {centered_corr(true_quantiles[:, i], preds[:, i]):.4f}"
        )

    if "intercept" in learned:
        print("\nIntercept check:")
        print(f"  true intercept quantiles    : {np.round(true_intercept_q, 4)}")
        print(f"  learned intercept quantiles : {np.round(learned['intercept'], 4)}")

    # ------------------------------------------------------------------
    # Category-level grouped summaries
    # ------------------------------------------------------------------
    if learned_region_q is not None:
        print("\nMean learned region contribution by category:")
        for region in sorted(REGION_MU.keys()):
            mask = xv["region"].to_numpy() == region
            means = learned_region_q[mask].mean(axis=0)
            print(
                f"  {region:>5s} | learned means per q: "
                f"{np.round(means, 4).tolist()}"
            )

    # ------------------------------------------------------------------
    # Optional plot: true vs predicted median
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        true_quantiles[:, 1],
        preds[:, 1],
        alpha=0.5,
        s=18,
    )
    lo = min(true_quantiles[:, 1].min(), preds[:, 1].min())
    hi = max(true_quantiles[:, 1].max(), preds[:, 1].max())
    ax.plot([lo, hi], [lo, hi], "--")
    ax.set_xlabel("True conditional median")
    ax.set_ylabel("Predicted median")
    ax.set_title("QNAM verification: median prediction")

    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "example_qnam_plot.png",
    )
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out}")

    print("\nInterpretation:")
    print("  - MAE vs true quantiles should be reasonably small.")
    print("  - Correlations should be clearly positive, often high.")
    print("  - Non-crossing prediction rate should be 1.0 (or extremely close).")
    print("  - Feature contributions should also have nonnegative adjacent quantile gaps.")
    print("  - region[...] keys confirm one-hot categorical effects are handled atomically.")


if __name__ == "__main__":
    main()