#!/usr/bin/env python3
"""
Verification example for NBMRegressor with one-hot categorical features.

Checks:
- returned contribution keys include region[0], region[1], ...
- summed one-hot contributions recover the true categorical effect
- grouped means by category line up with the data-generating process

Run:
    python examples/example_nbm_onehot.py
"""

import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from nampy.models import NBMRegressor


def main():
    rng = np.random.default_rng(123)
    n = 1000

    # ------------------------------------------------------------------
    # Mixed data: numerical + categorical
    # ------------------------------------------------------------------
    X = pd.DataFrame({
        "age": rng.uniform(18, 70, n),
        "income": rng.lognormal(mean=10.0, sigma=0.6, size=n),
        "region": rng.choice(["North", "South", "East", "West"], size=n, p=[0.25, 0.25, 0.25, 0.25]),
        "channel": rng.choice(["Online", "Retail", "Partner"], size=n, p=[0.4, 0.4, 0.2]),
    })

    REGION_EFFECT = {
        "North": -0.35,
        "South": 0.10,
        "East": 0.00,
        "West": 0.25,
    }

    CHANNEL_EFFECT = {
        "Online": 0.20,
        "Retail": -0.10,
        "Partner": 0.05,
    }

    def f_age(x):
        return 0.025 * (x - x.mean())

    def f_income(x):
        z = np.log(x)
        return 0.35 * (z - z.mean()) / z.std()

    true_age = f_age(X["age"].to_numpy())
    true_income = f_income(X["income"].to_numpy())
    true_region = np.array([REGION_EFFECT[r] for r in X["region"]])
    true_channel = np.array([CHANNEL_EFFECT[c] for c in X["channel"]])

    noise = rng.normal(0.0, 0.25, n)
    y = true_age + true_income + true_region + true_channel + noise

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------------------------------------------------------
    # NBM model
    #
    # - unary only, because this test is about categorical one-hot features
    # - one-hot categoricals so returned keys should include region[0], ...
    # ------------------------------------------------------------------
    model = NBMRegressor(
        layer_sizes=(128, 64),
        num_bases=64,
        num_subnets=1,
        dropout_rate=0.05,
        bases_dropout=0.05,
        output_penalty=1e-4,
        nary=[1],
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

    preds = model.predict(X_val)
    scores = model.evaluate(
        X_val, y_val,
        metrics={"MAE": mean_absolute_error, "R2": r2_score},
    )
    print(f"Validation — MAE: {scores['MAE']:.4f}, R2: {scores['R2']:.4f}")

    feat_vals = model.predict_feature_vals(X_val)

    print("\nReturned contribution keys:")
    print(list(feat_vals.keys()))

    def to_numpy(v):
        t = getattr(v, "detach", lambda: v)()
        arr = np.asarray(t)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        return arr.reshape(-1)

    learned = {
        k: to_numpy(v)
        for k, v in feat_vals.items()
        if k not in {"output", "intercept", "output_penalty"}
    }

    # ------------------------------------------------------------------
    # Check one-hot key presence
    # ------------------------------------------------------------------
    region_keys = sorted([k for k in learned if k.startswith("region[")])
    channel_keys = sorted([k for k in learned if k.startswith("channel[")])

    print("\nOne-hot categorical keys found:")
    print(f"  region keys : {region_keys}")
    print(f"  channel keys: {channel_keys}")

    if not region_keys:
        raise RuntimeError("No region[...] keys were returned. One-hot handling may be broken.")
    if not channel_keys:
        raise RuntimeError("No channel[...] keys were returned. One-hot handling may be broken.")

    # ------------------------------------------------------------------
    # Build validation-set truth
    # ------------------------------------------------------------------
    xv = X_val.reset_index(drop=True)

    true_age_val = f_age(xv["age"].to_numpy())
    true_income_val = f_income(xv["income"].to_numpy())
    true_region_val = np.array([REGION_EFFECT[r] for r in xv["region"]])
    true_channel_val = np.array([CHANNEL_EFFECT[c] for c in xv["channel"]])
    true_total_no_noise = (
        true_age_val + true_income_val + true_region_val + true_channel_val
    )

    # ------------------------------------------------------------------
    # Aggregate one-hot atomic terms back to logical categorical effects
    # ------------------------------------------------------------------
    learned_region_total = sum(learned[k] for k in region_keys)
    learned_channel_total = sum(learned[k] for k in channel_keys)

    def centered_corr(a, b):
        a = np.asarray(a).reshape(-1)
        b = np.asarray(b).reshape(-1)
        a = a - a.mean()
        b = b - b.mean()
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return np.nan
        return np.corrcoef(a, b)[0, 1]

    print("\nVerification (true DGP vs learned contributions):")
    print(f"  total    true vs predictions correlation: {centered_corr(true_total_no_noise, preds):.4f}")

    if "age" in learned:
        print(f"  age      true vs learned correlation:     {centered_corr(true_age_val, learned['age']):.4f}")
    else:
        print("  age      key not found")

    if "income" in learned:
        print(f"  income   true vs learned correlation:     {centered_corr(true_income_val, learned['income']):.4f}")
    else:
        print("  income   key not found")

    print(f"  region   true vs learned correlation:     {centered_corr(true_region_val, learned_region_total):.4f}")
    print(f"  channel  true vs learned correlation:     {centered_corr(true_channel_val, learned_channel_total):.4f}")

    # ------------------------------------------------------------------
    # Category-level sanity check:
    # mean learned categorical contribution within each category
    # should align with the true category effect ordering
    # ------------------------------------------------------------------
    print("\nMean learned region contribution by category:")
    region_summary = []
    for region in sorted(REGION_EFFECT.keys()):
        mask = xv["region"].to_numpy() == region
        mean_learned = learned_region_total[mask].mean()
        region_summary.append((region, REGION_EFFECT[region], mean_learned))
        print(f"  {region:>5s} | true: {REGION_EFFECT[region]: .3f} | learned mean: {mean_learned: .3f}")

    print("\nMean learned channel contribution by category:")
    channel_summary = []
    for channel in sorted(CHANNEL_EFFECT.keys()):
        mask = xv["channel"].to_numpy() == channel
        mean_learned = learned_channel_total[mask].mean()
        channel_summary.append((channel, CHANNEL_EFFECT[channel], mean_learned))
        print(f"  {channel:>7s} | true: {CHANNEL_EFFECT[channel]: .3f} | learned mean: {mean_learned: .3f}")

    # ------------------------------------------------------------------
    # Optional: inspect individual region[...] terms
    # This does NOT identify which index corresponds to which category;
    # it just confirms the atomic one-hot terms are active and nontrivial.
    # ------------------------------------------------------------------
    print("\nMean absolute contribution of each one-hot atomic region term:")
    for k in region_keys:
        print(f"  {k:10s}: {np.mean(np.abs(learned[k])):.4f}")

    print("\nMean absolute contribution of each one-hot atomic channel term:")
    for k in channel_keys:
        print(f"  {k:10s}: {np.mean(np.abs(learned[k])):.4f}")

    print("\nInterpretation:")
    print("  - You should see keys like region[0], region[1], ... and channel[0], ...")
    print("  - The summed region[...] contribution should correlate strongly with the true region effect.")
    print("  - The grouped means by category should preserve the correct ordering of effects.")
    print("  - Individual region[k] terms need not be directly human-readable unless you also inspect encoder category order.")

    # ------------------------------------------------------------------
    # Optional plotting through your package API
    # ------------------------------------------------------------------
    try:
        fig = model.plot(X_val, y_val)
        if fig is not None:
            out = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "example_nbm_onehot_plot.png",
            )
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"\nPlot saved: {out}")
    except Exception as e:
        print(f"\nPlotting skipped: {e}")


if __name__ == "__main__":
    main()