"""Reproduce the new-feature guide figures from independently split fitted data.

Run from a repository checkout with ``python examples/example_capability_figures.py``.
The three tutorial notebooks contain the corresponding standalone analyses.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import beta as beta_distribution

from nampy import (
    GAMLSS,
    CalibratedQuantileGAM,
    QuantileGAM,
    SemiStructuredRegressor,
    SemiStructuredTrainingConfig,
    SurvivalGAM,
)
from nampy.calibration import calibrate_regressor
from nampy.neural.configs import DefaultSemiStructuredConfig
from nampy.survival import (
    CensoringDistribution,
    SurvivalTarget,
    brier_score,
    concordance_index,
)

FIGURES = Path(__file__).resolve().parents[1] / "docs" / "_static" / "features"


def show_figure(figure, name):
    """Persist publication-independent guide assets, then close pyplot state."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def survival_example():
    """Model event times with a Weibull likelihood and evaluate unseen subjects."""
    rng = np.random.default_rng(73)
    X = pd.DataFrame({"x": rng.uniform(-1, 1, 480)})
    event_time = np.exp(0.6 - 0.8 * X.x.to_numpy()) * rng.weibull(1.6, len(X))
    censoring_time = rng.exponential(4.0, len(X))
    target = SurvivalTarget(
        np.minimum(event_time, censoring_time), event_time <= censoring_time,
        subject_id=np.arange(len(X)),
    )
    # Each row is one subject. Randomize subjects before any model fitting.
    order = rng.permutation(len(X))
    train, test = order[:320], order[320:]
    y_train, y_test = target.take(train), target.take(test)
    model = SurvivalGAM(
        likelihood="weibull", formula={"scale": "~x", "shape": "~1"},
        optimize_smoothing=False,
    ).fit(X.iloc[train], y_train)
    prediction = model.predict_distribution(X.iloc[test])
    censoring = CensoringDistribution.fit(y_train)
    horizons = np.array([0.5, 1.0, 1.5, 2.0])
    brier = brier_score(prediction, y_test, horizons, censoring=censoring)
    concordance = concordance_index(
        y_test, -prediction.rmst(2.0), higher_risk=True
    )
    profiles = pd.DataFrame({"x": [-0.8, 0.0, 0.8]})
    curves = model.predict_distribution(profiles)
    times = np.linspace(0.02, 3, 180)
    colors = ["#276A9D", "#D28C2D", "#A24669"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    rows = test[:28]
    rows = rows[np.argsort(target.time[rows])]
    for number, row in enumerate(rows):
        axes[0].plot([0, target.time[row]], [number, number], color="0.75", lw=1)
        axes[0].plot(target.time[row], number, "o" if target.event[row] else "+",
                     color="#276A9D" if target.event[row] else "#A24669", ms=5)
    axes[0].plot([], [], "o", color="#276A9D", label="Observed event")
    axes[0].plot([], [], "+", color="#A24669", label="Right censored")
    axes[0].set(xlabel="Observed follow-up (arbitrary units)", ylabel="Held-out subject",
                title="28 held-out follow-up records")
    axes[0].legend(fontsize=8)
    for row, color in enumerate(colors):
        axes[1].plot(times, curves.survival(times)[row], color=color,
                     label=f"x = {profiles.x.iloc[row]:.1f}")
    axes[1].set(xlabel="Time (arbitrary units)", ylabel="P(T > time | x)", ylim=(0, 1.02),
                title="Fitted survival for three profiles")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    show_figure(figure, "survival_profiles")
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for row, color in enumerate(colors):
        axes[0].plot(times, curves.hazard(times)[row], color=color,
                     label=f"x = {profiles.x.iloc[row]:.1f}")
        axes[1].plot(times, curves.survival(times)[row], color=color)
    axes[0].set(xlabel="Time (arbitrary units)", ylabel="Events per arbitrary time unit",
                title="Hazard among subjects still at risk")
    axes[1].set(xlabel="Time (arbitrary units)", ylabel="P(T > time | x)", ylim=(0, 1.02),
                title="Probability of surviving beyond each time")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    show_figure(figure, "survival_hazard")
    results = {
        "training_subjects": len(train), "test_subjects": len(test),
        "test_censored_fraction": float(np.mean(~y_test.event)),
        "brier": brier.score.tolist(), "integrated_brier": brier.integrated(),
        "harrell_concordance": concordance.concordance,
        "profile_survival_at_2": curves.survival([2.0])[:, 0].tolist(),
    }
    print(
        f"Survival: {len(train)} training and {len(test)} test subjects; "
        f"integrated IPCW Brier error {brier.integrated():.3f}, "
        f"Harrell concordance {concordance.concordance:.3f}."
    )
    return results


def quantile_example():
    """Separate uncertainty in a median from variation in a future response."""
    rng = np.random.default_rng(82)
    X = pd.DataFrame({"x": rng.uniform(-1.5, 1.5, 600)})
    y = 0.4 + 0.8 * X.x.to_numpy() + rng.normal(0, 0.65, len(X))
    order = rng.permutation(len(X))
    train, calibration, test = order[:300], order[300:450], order[450:]
    formula = "~s(x,k=5)"
    quantile = QuantileGAM(
        formula=formula, level=0.5, smoothing_params=[0.5], bandwidth=0.2, tol=1e-7
    ).fit(X.iloc[train], y[train])
    assert quantile.fit_result_.converged
    calibrated = CalibratedQuantileGAM(
        formula=formula, level=0.5, bandwidth=0.2, smoothing_params=[0.5],
        log_scale_bounds=(-4, 2),
    ).fit(X.iloc[train], y[train])
    # This separate cohort calibrates future-response coverage; it does not
    # train either quantile model or calibrate its coefficient uncertainty.
    response = calibrate_regressor(
        quantile, X.iloc[calibration], y[calibration], level=0.9
    )
    grid = pd.DataFrame({"x": np.linspace(-1.5, 1.5, 160)})
    confidence = calibrated.predict_interval(grid, level=0.95)
    future = response.predict_interval(grid)
    test_interval = response.predict_interval(X.iloc[test])
    coverage = np.mean((y[test] >= test_interval.lower) & (y[test] <= test_interval.upper))
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True)
    for axis in axes:
        axis.scatter(X.x.iloc[test], y[test], s=9, color="0.6", alpha=0.45,
                     label="Held-out responses")
        axis.plot(grid.x, 0.4 + 0.8 * grid.x, "k--", lw=1, label="True median")
    axes[0].fill_between(grid.x, confidence.lower, confidence.upper,
                         color="#276A9D", alpha=0.25, label="95% pointwise confidence interval")
    axes[0].plot(grid.x, confidence.estimate, color="#276A9D")
    axes[1].fill_between(grid.x, future.lower, future.upper, color="#D28C2D",
                         alpha=0.25, label="90% future-response interval")
    axes[1].plot(grid.x, quantile.predict(grid), color="#D28C2D")
    axes[0].set(title="Uncertainty about the conditional median", xlabel="x", ylabel="Response")
    axes[1].set(title="Variation in a future response", xlabel="x")
    for axis in axes:
        axis.legend(fontsize=7, loc="upper left")
    figure.tight_layout()
    show_figure(figure, "quantile_targets")
    # A deliberately narrow, uncalibrated response band has no coverage claim.
    # Report both its width and independent-test coverage beside conformal's.
    before = quantile.predict(X.iloc[test])
    baseline_coverage = np.mean(np.abs(y[test] - before) <= 0.5)
    widths = [1.0, float(np.mean(test_interval.upper - test_interval.lower))]
    figure, axes = plt.subplots(
        1, 3, figsize=(12, 3.7), gridspec_kw={"width_ratios": [1.5, 1, 1]}
    )
    axes[0].fill_between(grid.x, quantile.predict(grid) - 0.5,
                         quantile.predict(grid) + 0.5, color="0.65", alpha=0.4,
                         label="Uncalibrated ±0.5 response band")
    axes[0].plot(grid.x, future.lower, color="#276A9D", label="90% conformal response bounds")
    axes[0].plot(grid.x, future.upper, color="#276A9D")
    axes[0].scatter(X.x.iloc[test], y[test], s=8, color="0.4", alpha=0.35)
    axes[0].set(xlabel="x", ylabel="Response", title="Calibration changes the response band")
    axes[0].legend(fontsize=7)
    positions = np.arange(2)
    labels = ["Uncalibrated", "Conformal"]
    axes[1].bar(positions, [baseline_coverage, coverage], color="#276A9D", width=0.6)
    axes[1].axhline(0.9, color="0.35", ls="--", lw=1, label="90% target")
    axes[1].set(xticks=positions, xticklabels=labels, ylabel="Fraction covered",
                ylim=(0, 1.05), title="Held-out coverage")
    axes[1].legend(fontsize=7)
    axes[2].bar(positions, widths, color="#D28C2D", width=0.6)
    axes[2].set(xticks=positions, xticklabels=labels, ylabel="Response units",
                ylim=(0, max(widths) * 1.25), title="Mean interval width")
    figure.tight_layout()
    show_figure(figure, "conformal_intervals")
    results = {
        "train_calibration_test": [len(train), len(calibration), len(test)],
        "quantile_function_mean_width": float(np.mean(confidence.upper - confidence.lower)),
        "conformal_coverage": float(coverage), "conformal_width": widths[1],
        "uncalibrated_coverage": float(baseline_coverage), "uncalibrated_width": widths[0],
        "loss_scale": calibrated.loss_scale_,
        "quantile_function_coverage_metadata": confidence.coverage,
    }
    print(
        f"Quantiles: pointwise median CI mean width {results['quantile_function_mean_width']:.3f}; "
        f"conformal test coverage {coverage:.1%}, response interval width {widths[1]:.3f}."
    )
    return results


def semistructured_example():
    """Fit the full-cohort projection and reuse it for every prediction batch."""
    rng = np.random.default_rng(67)
    X = pd.DataFrame({"x": rng.uniform(-1, 1, 240), "z": rng.uniform(-1, 1, 240)})
    y = 0.5 + np.sin(3 * X.x.to_numpy()) + 1.3 * X.x.to_numpy() * X.z.to_numpy()
    y += rng.normal(0, 0.1, len(X))
    train, validation, test = np.arange(160), np.arange(160, 200), np.arange(200, 240)
    architecture = DefaultSemiStructuredConfig(
        structured_formula="~s(x,k=5)+z", neural_features=("x", "z"),
        hidden_dims=(12,), representation_dim=5,
    )
    model = SemiStructuredRegressor(
        architecture_config=architecture,
        training=SemiStructuredTrainingConfig(
            epochs=120, lr=0.025, patience=25, smoothing_params=[0.2], random_state=18,
        ),
    ).fit(X.iloc[train], y[train], X_val=X.iloc[validation], y_val=y[validation])
    grid = pd.DataFrame({"x": np.linspace(-1, 1, 120), "z": np.full(120, 0.8)})
    components = model.predict_components(grid)
    components.validate_additive_reconstruction()
    neural = components.terms["neural_component"][:, 0]
    structured = components.link[:, 0] - neural
    # Query rows are not used to compute a fresh projection. A singleton and
    # the same row inside a batch must have identical fitted predictions.
    np.testing.assert_allclose(model.predict(grid.iloc[:1]), model.predict(grid)[:1], atol=1e-12)
    figure, axis = plt.subplots(figsize=(7, 3.8))
    axis.plot(grid.x, structured, color="#276A9D", label="Structured part (including intercept)")
    axis.plot(grid.x, neural, color="#D28C2D", label="Projected neural remainder")
    axis.plot(grid.x, components.link[:, 0], color="#A24669", lw=2, label="Total prediction")
    axis.plot(grid.x, 0.5 + np.sin(3 * grid.x) + 1.3 * grid.x * grid.z,
              "k--", lw=1, label="True mean on z=0.8 slice")
    axis.set(xlabel="x (z held at 0.8)", ylabel="Contribution / predicted response",
              title="Structured and neural contributions at z = 0.8")
    axis.legend(fontsize=8)
    figure.tight_layout()
    show_figure(figure, "semistructured_components")
    results = {
        "training_validation_test": [len(train), len(validation), len(test)],
        "training_orthogonality_error": model.orthogonality_error_,
        "test_rmse": float(np.sqrt(np.mean((model.predict(X.iloc[test]).ravel() - y[test]) ** 2))),
        "neural_slice_range": [float(neural.min()), float(neural.max())],
        "batch_invariance_max_error": float(np.max(np.abs(model.predict(grid.iloc[:1]) - model.predict(grid)[:1]))),
    }
    print(
        f"Semi-structured: held-out RMSE {results['test_rmse']:.3f}; "
        f"neural contribution range {neural.min():.2f} to {neural.max():.2f}."
    )
    return results


def distribution_examples():
    """Display fitted conditional laws without conflating density and atom mass."""
    rng = np.random.default_rng(44)
    X = pd.DataFrame({"x": rng.uniform(-1, 1, 500)})
    loc = 0.3 + 0.7 * X.x.to_numpy()
    scale = np.exp(-0.8 + 0.55 * X.x.to_numpy())
    y = rng.lognormal(loc, scale)
    model = GAMLSS(
        family="lognormal", formula={"loc": "~x", "scale": "~x"},
        optimize_smoothing=False,
    ).fit(X, y)
    profiles = pd.DataFrame({"x": [-0.8, 0.0, 0.8]})
    prediction = model.predict_distribution(profiles)
    parameters = model.predict_parameters(profiles)
    grid = np.linspace(0.01, 7, 350)
    figure, axis = plt.subplots(figsize=(7, 3.7))
    for row, color in enumerate(["#276A9D", "#D28C2D", "#A24669"]):
        density = np.array([np.exp(prediction.log_prob(np.full(3, value)))[row] for value in grid])
        axis.plot(grid, density, color=color, label=f"x = {profiles.x.iloc[row]:.1f}")
        axis.axvline(prediction.quantile([0.5])[row, 0], color=color, ls=":", lw=1)
    axis.set(xlabel="Positive response", ylabel="Conditional density",
              title="Location, spread and skewness change with x")
    axis.plot([], [], ":", color="0.4", label="Dotted lines: conditional medians")
    axis.legend(fontsize=8)
    figure.tight_layout()
    show_figure(figure, "distribution_slices")
    # Coupled softmax atom probabilities plus a continuous beta interior.
    x = X.x.to_numpy()
    logits = np.column_stack([-1.0 - x, -1.5 + x, np.zeros(len(x))])
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    mu, phi = expit(0.2 + 0.6 * x), 14.0
    choice = np.array([rng.choice(3, p=p) for p in weights])
    y = rng.beta(mu * phi, (1 - mu) * phi)
    y[choice == 0], y[choice == 1] = 0.0, 1.0
    mixed = GAMLSS(
        family="beinf", formula={"mu": "~x", "phi": "~1", "pi0": "~x", "pi1": "~x"},
        optimize_smoothing=False,
    ).fit(X, y)
    natural = mixed.predict_parameters(profiles)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    positions = np.arange(3)
    axes[0].bar(positions - 0.2, natural[:, 2], 0.2, label="P(Y=0)", color="#276A9D")
    axes[0].bar(positions, natural[:, 3], 0.2, label="P(Y=1)", color="#A24669")
    axes[0].bar(positions + 0.2, natural[:, 4], 0.2, label="P(0<Y<1)", color="#D28C2D")
    axes[0].set(xticks=positions, xticklabels=["x=-0.8", "x=0", "x=0.8"],
                ylabel="Probability mass", ylim=(0, 1), title="Separate endpoint and interior probabilities")
    axes[0].legend(fontsize=7)
    interior = np.linspace(0.001, 0.999, 250)
    for row, color in enumerate(["#276A9D", "#D28C2D", "#A24669"]):
        m, precision, _, _, mass = natural[row]
        axes[1].plot(interior, mass * beta_distribution.pdf(interior, m * precision, (1-m) * precision),
                     color=color, label=f"x = {profiles.x.iloc[row]:.1f}")
    axes[1].set(xlabel="Interior response", ylabel="Unconditional continuous density",
                title="Each curve integrates to its interior probability")
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    show_figure(figure, "inflated_components")
    results = {"lognormal_parameters": parameters.tolist(), "mixed_parameters": natural.tolist()}
    print("Saved fitted lognormal density slices and inflated-beta component probabilities.")
    return results


if __name__ == "__main__":
    import json

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    results = {
        "survival": survival_example(), "quantile": quantile_example(),
        "semistructured": semistructured_example(), "distributions": distribution_examples(),
    }
    (FIGURES / "results.json").write_text(json.dumps(results, indent=2) + "\n")
