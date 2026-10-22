"""Generate deterministic figures used by the documentation tutorials."""

from pathlib import Path
import logging
import sys
import tempfile
import warnings


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPOSITORY_ROOT / "docs"
sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from nampy.gam import GAM
from nampy.models import GAMRegressor, NAMRegressor, NAMLSS, NBMRegressor


PINK = "#b4234d"
PINK_FILL = "#f3b4c2"
BLUE = "#276fbf"


def _save(figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(
        DOCS_ROOT / "_static" / name,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def make_data() -> pd.DataFrame:
    """Return the deterministic synthetic energy-demand dataset."""
    rng = np.random.default_rng(21)
    n = 900
    temperature = rng.uniform(-5.0, 35.0, n)
    hour = rng.uniform(0.0, 24.0, n)
    demand = (
        45.0
        + 0.12 * (temperature - 18.0) ** 2
        + 5.0 * np.sin(2.0 * np.pi * (hour - 7.0) / 24.0)
        + rng.normal(0.0, 2.0, n)
    )
    return pd.DataFrame(
        {"demand": demand, "temperature": temperature, "hour": hour}
    )


def generate_first_gam_plot() -> None:
    train, _ = train_test_split(make_data(), test_size=0.2, random_state=21)
    model = GAMRegressor(
        formula=(
            'demand ~ s(temperature, bs="cr", k=12) '
            '+ s(hour, bs="cc", k=12)'
        ),
        family="gaussian",
    ).fit(train)

    rendered = model.plot(
        pages=1,
        rug=False,
        scheme=1,
        shade_col=PINK_FILL,
        scale=0,
        figsize=(10, 4),
    )
    figure = rendered["figures"][0]
    _save(figure, "first_gam_effects.png")

    fitted = model.predict(train)
    residual = model.gam_.residuals(type="deviance")
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.scatter(fitted, residual, s=15, color="#6b7280", alpha=0.5)
    axis.axhline(0.0, color=PINK, linewidth=1.5)
    axis.set(xlabel="Fitted demand", ylabel="Deviance residual")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.16)
    _save(figure, "first_gam_residuals.png")


def generate_shape_constraint_plot() -> None:
    """Compare an ordinary smooth with a monotone-increasing smooth."""
    rng = np.random.default_rng(9)
    x = np.sort(rng.uniform(0.0, 10.0, 180))
    signal = 1.0 + 4.0 / (1.0 + np.exp(-(x - 5.0)))
    data = pd.DataFrame({"y": signal + rng.normal(0.0, 0.45, x.size), "x": x})

    ordinary = GAM(
        formula='y ~ s(x, bs="cr", k=12)',
        family="gaussian",
        optimize_smoothing=True,
        smoothing_method="reml",
    ).fit(data)
    increasing = GAM(
        formula='y ~ s(x, bs="mpi", k=12)',
        family="gaussian",
        optimize_smoothing=True,
        smoothing_method="gcv",
        smoothing_optimizer="bfgs",
    ).fit(data)

    grid = pd.DataFrame({"x": np.linspace(0.0, 10.0, 240)})
    ordinary_fit = ordinary.predict(grid, type="response")
    constrained_fit, constrained_se = increasing.predict(
        grid, type="response", return_se=True
    )

    figure, axis = plt.subplots(figsize=(8.4, 4.6))
    axis.scatter(data["x"], data["y"], s=13, color="#9ca3af", alpha=0.48)
    axis.plot(grid["x"], ordinary_fit, color="#374151", lw=2, label="Unconstrained")
    axis.fill_between(
        grid["x"],
        constrained_fit - 2.0 * constrained_se,
        constrained_fit + 2.0 * constrained_se,
        color=PINK_FILL,
        alpha=0.72,
        linewidth=0,
    )
    axis.plot(
        grid["x"],
        constrained_fit,
        color=PINK,
        lw=2.4,
        label="Monotone increasing",
    )
    axis.set(xlabel="x", ylabel="Expected response")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    _save(figure, "shape_constraint_comparison.png")


def generate_interaction_plot() -> None:
    """Fit and render a tensor-product interaction through ``GAM.plot``."""
    rng = np.random.default_rng(33)
    n = 700
    latitude = rng.uniform(-1.0, 1.0, n)
    longitude = rng.uniform(-1.0, 1.0, n)
    response = (
        2.5 * np.exp(-4.0 * ((latitude - 0.25) ** 2 + (longitude + 0.2) ** 2))
        - 1.6 * np.exp(-5.0 * ((latitude + 0.45) ** 2 + (longitude - 0.4) ** 2))
        + rng.normal(0.0, 0.16, n)
    )
    data = pd.DataFrame(
        {"response": response, "latitude": latitude, "longitude": longitude}
    )
    model = GAM(
        formula="response ~ te(latitude, longitude, bs=c('cr', 'cr'), k=c(8, 8))",
        family="gaussian",
        optimize_smoothing=True,
        smoothing_method="reml",
    ).fit(data)
    rendered = model.plot(pages=1, scheme=2, rug=False, figsize=(7.2, 5.0))
    figure = rendered["figures"][0]
    figure.axes[0].set_title("Joint spatial effect")
    _save(figure, "interaction_surface.png")


def _neural_fit_kwargs(checkpoint_path: Path) -> dict:
    return {
        "max_epochs": 30,
        "lr": 1e-2,
        "batch_size": 128,
        "random_state": 17,
        "checkpoint_path": checkpoint_path,
        "logger": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "num_sanity_val_steps": 0,
    }


def generate_neural_comparison() -> None:
    """Compare NAM and NBM effects using their public explanation tables."""
    rng = np.random.default_rng(17)
    n = 700
    temperature = rng.uniform(-2.0, 2.0, n)
    humidity = rng.uniform(-2.0, 2.0, n)
    response = (
        1.7 * np.sin(1.5 * temperature)
        + 0.65 * humidity**2
        + rng.normal(0.0, 0.18, n)
    )
    X = pd.DataFrame({"temperature": temperature, "humidity": humidity})

    with tempfile.TemporaryDirectory(prefix="nampy-docs-neural-") as directory:
        checkpoint = Path(directory)
        fit_kwargs = _neural_fit_kwargs(checkpoint)
        models = {
            "NAM": NAMRegressor(
                layer_sizes=[24, 12], dropout=0.0,
                numerical_preprocessing="minmax",
            ),
            "NBM": NBMRegressor(
                layer_sizes=[16], num_bases=8, nary=[1], dropout=0.0,
                bases_dropout=0.0, numerical_preprocessing="minmax",
            ),
        }
        tables = {}
        for name, model in models.items():
            model.fit(X, response, **fit_kwargs)
            model.predict_components(X, center=True).validate_additive_reconstruction(
                atol=1e-5
            )
            tables[name] = model.explain_terms(X, center=True, max_bins=28)

    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), sharey="col")
    features = ("temperature", "humidity")
    for row, (name, table) in enumerate(tables.items()):
        for column, feature in enumerate(features):
            axis = axes[row, column]
            term = table[table["term"] == feature].sort_values("value")
            axis.plot(
                term["value"], term["contribution"], color=PINK, linewidth=2.2
            )
            axis.set_title(f"{name}: {feature}")
            axis.set_xlabel("Feature value")
            if column == 0:
                axis.set_ylabel("Centered contribution")
            axis.grid(alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)
    _save(figure, "nam_nbm_comparison.png")


def generate_distributional_plot() -> None:
    """Fit varying location and scale and call the public effect renderer."""
    rng = np.random.default_rng(29)
    n = 800
    exposure = rng.uniform(0.0, 1.0, n)
    seasonal_index = rng.uniform(-1.0, 1.0, n)
    location = 1.2 * np.sin(2.0 * np.pi * exposure) + 0.35 * seasonal_index
    spread = 0.18 + 0.7 * exposure
    response = location + spread * rng.normal(size=n)
    X = pd.DataFrame(
        {"exposure": exposure, "seasonal_index": seasonal_index}
    )

    with tempfile.TemporaryDirectory(prefix="nampy-docs-lss-") as directory:
        model = NAMLSS(
            family="normal",
            layer_sizes=[24, 12],
            dropout=0.0,
            numerical_preprocessing="minmax",
        )
        model.fit(X, response, **_neural_fit_kwargs(Path(directory)))
        figures = model.plot_distributional_effects(
            X,
            reference_X=X,
            feature_names=["exposure"],
            point_label="conditional location effect",
            envelope_label="location effect ± conditional SD",
            ylabel="Response-scale effect and spread",
            title="Location and conditional spread change with exposure",
            ncols=1,
            figsize=(7.4, 4.4),
        )
    _save(figures[0], "changing_variability.png")


def main() -> None:
    warnings.filterwarnings("ignore")
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    generate_first_gam_plot()
    generate_shape_constraint_plot()
    generate_interaction_plot()
    generate_neural_comparison()
    generate_distributional_plot()


if __name__ == "__main__":
    main()
