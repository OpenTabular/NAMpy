#!/usr/bin/env python3
"""
Verification example for GAMClassifier on synthetic additive binary data.

What this checks
----------------
1. Predictive quality on a smooth additive logistic signal.
2. Exact additive decomposition on the LINK scale:
      output == intercept + sum(returned feature contributions)
3. Probability consistency:
      response == sigmoid(output)
4. Recovery of smooth numeric effects.

Run:
    python examples/example_gam_classifier.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from nampy.models import GAMClassifier


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def main():
    rng = np.random.default_rng(42)
    n = 1800

    # ------------------------------------------------------------------
    # Numeric-only data (phase-1 GAM currently supports numeric features only)
    # ------------------------------------------------------------------
    X = pd.DataFrame(
        {
            "age": rng.uniform(18, 70, n),
            "income": rng.uniform(20, 140, n),
        }
    )

    # ------------------------------------------------------------------
    # Smooth additive logistic data-generating process
    # ------------------------------------------------------------------
    age_raw = X["age"].to_numpy()
    income_raw = X["income"].to_numpy()

    age_s = (age_raw - age_raw.mean()) / age_raw.std()
    income_s = (income_raw - income_raw.mean()) / income_raw.std()

    def f_age(x):
        return 1.1 * np.sin(1.7 * x) + 0.25 * x

    def f_income(x):
        return 0.9 * x - 0.55 * (x**2)

    intercept_true = -0.15

    true_age = f_age(age_s)
    true_income = f_income(income_s)
    true_eta = intercept_true + true_age + true_income
    true_prob = sigmoid(true_eta)

    y = rng.binomial(1, true_prob, size=n)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------------------------------------------
    # GAM classifier: automatic REML smoothing selection (the default)
    # ------------------------------------------------------------------
    model = GAMClassifier(
        family="binomial",
        k=12,
        fit_intercept=True,
    )

    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    proba = model.predict_proba(X_val)[:, 1]
    preds = model.predict(X_val)
    eta_pred = model.decision_function(X_val)

    acc = accuracy_score(y_val, preds)
    ll = log_loss(y_val, np.clip(proba, 1e-9, 1.0 - 1e-9))
    auc = roc_auc_score(y_val, proba)

    print(f"Validation — Accuracy: {acc:.4f}, LogLoss: {ll:.4f}, AUROC: {auc:.4f}")

    # ------------------------------------------------------------------
    # Returned additive contributions
    # ------------------------------------------------------------------
    feat_vals = model.predict_feature_vals(X_val)
    print("\nReturned contribution keys:")
    print(list(feat_vals.keys()))

    def to_numpy(v):
        arr = np.asarray(v)
        if arr.ndim == 0:
            return float(arr)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        return arr.reshape(-1)

    learned = {k: to_numpy(v) for k, v in feat_vals.items()}

    # ------------------------------------------------------------------
    # Exact additive decomposition on LINK scale
    # ------------------------------------------------------------------
    reconstructed = np.zeros_like(eta_pred, dtype=float)

    if "intercept" in learned:
        reconstructed = reconstructed + float(learned["intercept"])

    for key in ["age", "income"]:
        if key in learned:
            reconstructed = reconstructed + learned[key]

    max_abs_link_error_1 = np.max(np.abs(reconstructed - learned["output"]))
    max_abs_link_error_2 = np.max(np.abs(reconstructed - eta_pred))

    print("\nExact additive decomposition check (link scale):")
    print(f"  max |reconstructed - returned output|     : {max_abs_link_error_1:.8f}")
    print(f"  max |reconstructed - decision_function()| : {max_abs_link_error_2:.8f}")

    # ------------------------------------------------------------------
    # Probability consistency
    # ------------------------------------------------------------------
    if "response" in learned:
        p_from_output = sigmoid(learned["output"])
        max_abs_prob_error = np.max(np.abs(p_from_output - learned["response"]))
        print("\nProbability consistency check:")
        print(f"  max |sigmoid(output) - response|          : {max_abs_prob_error:.8f}")

    # ------------------------------------------------------------------
    # Ground-truth contributions on validation set
    # ------------------------------------------------------------------
    xv = X_val.reset_index(drop=True)
    age_val_s = (xv["age"].to_numpy() - age_raw.mean()) / age_raw.std()
    income_val_s = (xv["income"].to_numpy() - income_raw.mean()) / income_raw.std()

    true_age_val = f_age(age_val_s)
    true_income_val = f_income(income_val_s)
    true_eta_val = intercept_true + true_age_val + true_income_val
    true_prob_val = sigmoid(true_eta_val)

    def centered_corr(a, b):
        a = np.asarray(a).reshape(-1)
        b = np.asarray(b).reshape(-1)
        a = a - a.mean()
        b = b - b.mean()
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return np.nan
        return np.corrcoef(a, b)[0, 1]

    print("\nVerification (true DGP vs learned effects):")
    print(
        f"  total link     true vs learned correlation: {centered_corr(true_eta_val, eta_pred):.4f}"
    )
    print(
        f"  total prob     true vs learned correlation: {centered_corr(true_prob_val, proba):.4f}"
    )

    if "age" in learned:
        print(
            f"  age effect     true vs learned correlation: {centered_corr(true_age_val, learned['age']):.4f}"
        )
    else:
        print("  age effect     key not found")

    if "income" in learned:
        print(
            f"  income effect  true vs learned correlation: {centered_corr(true_income_val, learned['income']):.4f}"
        )
    else:
        print("  income effect  key not found")

    # ------------------------------------------------------------------
    # Bin summaries for sanity-checking effect shapes
    # ------------------------------------------------------------------
    if "age" in learned:
        print("\nMean learned age effect by age quartile:")
        age_q = np.quantile(xv["age"], [0.25, 0.5, 0.75])
        age_bins = [
            ("Q1", xv["age"].to_numpy() < age_q[0]),
            (
                "Q2",
                (xv["age"].to_numpy() >= age_q[0]) & (xv["age"].to_numpy() < age_q[1]),
            ),
            (
                "Q3",
                (xv["age"].to_numpy() >= age_q[1]) & (xv["age"].to_numpy() < age_q[2]),
            ),
            ("Q4", xv["age"].to_numpy() >= age_q[2]),
        ]
        for name, mask in age_bins:
            print(
                f"  {name} | true mean: {true_age_val[mask].mean(): .3f} | "
                f"learned mean: {learned['age'][mask].mean(): .3f}"
            )

    if "income" in learned:
        print("\nMean learned income effect by income quartile:")
        inc_q = np.quantile(xv["income"], [0.25, 0.5, 0.75])
        inc_bins = [
            ("Q1", xv["income"].to_numpy() < inc_q[0]),
            (
                "Q2",
                (xv["income"].to_numpy() >= inc_q[0])
                & (xv["income"].to_numpy() < inc_q[1]),
            ),
            (
                "Q3",
                (xv["income"].to_numpy() >= inc_q[1])
                & (xv["income"].to_numpy() < inc_q[2]),
            ),
            ("Q4", xv["income"].to_numpy() >= inc_q[2]),
        ]
        for name, mask in inc_bins:
            print(
                f"  {name} | true mean: {true_income_val[mask].mean(): .3f} | "
                f"learned mean: {learned['income'][mask].mean(): .3f}"
            )

    # ------------------------------------------------------------------
    # Plot true vs learned probabilities
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    order_age = np.argsort(xv["age"].to_numpy())
    axes[0].plot(xv["age"].to_numpy()[order_age], true_age_val[order_age], label="true")
    if "age" in learned:
        axes[0].plot(
            xv["age"].to_numpy()[order_age], learned["age"][order_age], label="learned"
        )
    axes[0].set_title("Age effect")
    axes[0].set_xlabel("age")
    axes[0].set_ylabel("effect on log-odds")
    axes[0].legend()

    axes[1].scatter(true_prob_val, proba, alpha=0.5, s=18)
    lo = min(true_prob_val.min(), proba.min())
    hi = max(true_prob_val.max(), proba.max())
    axes[1].plot([lo, hi], [lo, hi], "--")
    axes[1].set_title("Predicted probability vs true probability")
    axes[1].set_xlabel("true probability")
    axes[1].set_ylabel("predicted probability")

    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "example_gam_classifier_plot.png",
    )
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out}")

    print("\nInterpretation:")
    print("  - Additive reconstruction error on the link scale should be ~0.")
    print("  - sigmoid(output) should match the returned response probabilities.")
    print("  - Accuracy / AUROC should be clearly better than chance.")
    print("  - age and income effect correlations should be clearly positive.")


if __name__ == "__main__":
    main()
