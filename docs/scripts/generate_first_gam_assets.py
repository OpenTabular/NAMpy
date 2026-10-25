"""Generate deterministic figures used by the documentation tutorials."""

import logging
import sys
import tempfile
import warnings
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPOSITORY_ROOT / "docs"
sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from nampy.gam import GAM  # noqa: E402
from nampy.models import NAMLSS, GAMRegressor, NAMRegressor, NBMRegressor  # noqa: E402

PINK = "#b4234d"
PINK_FILL = "#f3b4c2"
BLUE = "#276fbf"
GOLD = "#b7791f"
GREEN = "#2f855a"
GRAY = "#6b7280"


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


def make_gam_nam_data() -> pd.DataFrame:
    """Return the four-feature data used for the cross-backend comparison."""
    rng = np.random.default_rng(41)
    n = 1_200
    data = pd.DataFrame(
        {
            "median_income": rng.uniform(1.0, 10.0, n),
            "house_age": rng.uniform(1.0, 52.0, n),
            "latitude": rng.uniform(32.0, 42.0, n),
            "rooms_per_household": rng.lognormal(1.5, 0.35, n),
        }
    )
    data["value"] = (
        2.2 * np.tanh((data["median_income"] - 5.0) / 1.7)
        + 0.9 * np.sin((data["house_age"] - 5.0) / 8.0)
        - 0.55 * (data["latitude"] - 37.0) ** 2 / 6.0
        + 0.75 * np.log1p(data["rooms_per_household"])
        + rng.normal(0.0, 0.3, n)
    )
    return data


def generate_first_gam_plot() -> None:
    train, _ = train_test_split(make_data(), test_size=0.2, random_state=21)
    model = GAMRegressor(
        formula=(
            'demand ~ s(temperature, bs="cr", k=12) '
            '+ s(hour, bs="cc", k=12)'
        ),
        family="gaussian",
        knots={"hour": [0.0, 24.0]},
    ).fit(train)

    summary_path = DOCS_ROOT / "_generated/first_gam_summary.inc"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = str(model.summary())
    summary_path.write_text(
        ".. code-block:: text\n\n"
        + "".join(
            "   " + line.rstrip() + "\n" if line else "\n"
            for line in summary.splitlines()
        )
    )

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


def _component_for_feature(terms, feature: str) -> np.ndarray:
    """Select the unique named contribution involving ``feature``."""
    matches = [np.asarray(values) for name, values in terms.items() if feature in name]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one component for {feature!r}, found {len(matches)}."
        )
    return matches[0]


def generate_gam_nam_comparison() -> None:
    """Compare GAM and NAM contributions for every feature in one dataset."""
    data = make_gam_nam_data()
    train, test = train_test_split(data, test_size=0.2, random_state=41)
    features = (
        "median_income",
        "house_age",
        "latitude",
        "rooms_per_household",
    )

    gam = GAMRegressor(
        formula=(
            "value ~ s(median_income, bs='cr', k=12)"
            " + s(house_age, bs='cr', k=12)"
            " + s(latitude, bs='cr', k=12)"
            " + s(rooms_per_household, bs='cr', k=12)"
        ),
        family="gaussian",
    ).fit(train)

    with tempfile.TemporaryDirectory(prefix="nampy-docs-gam-nam-") as directory:
        nam = NAMRegressor(
            layer_sizes=[48, 24],
            dropout=0.0,
            numerical_preprocessing="minmax",
        )
        nam.fit(
            train.loc[:, list(features)],
            train["value"],
            max_epochs=60,
            lr=5e-3,
            batch_size=128,
            patience=10,
            random_state=41,
            checkpoint_path=Path(directory),
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )

    gam_training = gam.predict_components(train)
    nam_training = nam.predict_components(
        train.loc[:, list(features)],
        center=True,
        reference_X=train.loc[:, list(features)],
    )
    gam_training.validate_additive_reconstruction(atol=1e-7)
    nam_training.validate_additive_reconstruction(atol=1e-5)
    gam_r2 = gam.score(test, test["value"])
    nam_r2 = nam.score(test.loc[:, list(features)], test["value"])

    metrics_path = DOCS_ROOT / "_generated/gam_nam_scores.inc"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        ".. csv-table:: Test predictions\n   :header-rows: 1\n\n"
        f"   Model,Test R2\n   GAM,{gam_r2:.3f}\n   NAM,{nam_r2:.3f}\n"
    )

    display_names = {
        "median_income": "Median income",
        "house_age": "House age",
        "latitude": "Latitude",
        "rooms_per_household": "Rooms per household",
    }
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.0))
    for axis, feature in zip(axes.flat, features, strict=True):
        lower, upper = np.quantile(train[feature], [0.01, 0.99])
        x_values = np.linspace(lower, upper, 240)
        grid = pd.DataFrame(
            {
                name: np.full(x_values.size, train[name].median())
                for name in features
            }
        )
        grid[feature] = x_values

        gam_grid = gam.predict_components(grid)
        gam_effect = _component_for_feature(gam_grid.terms, feature)
        gam_reference = _component_for_feature(gam_training.terms, feature)
        gam_effect = gam_effect - np.mean(gam_reference, axis=0)

        nam_grid = nam.predict_components(
            grid,
            center=True,
            reference_X=train.loc[:, list(features)],
        )
        nam_effect = _component_for_feature(nam_grid.terms, feature)

        truth = {
            "median_income": lambda x: 2.2*np.tanh((x-5)/1.7),
            "house_age": lambda x: .9*np.sin((x-5)/8),
            "latitude": lambda x: -.55*(x-37)**2/6,
            "rooms_per_household": lambda x: .75*np.log1p(x),
        }[feature]
        axis.plot(x_values, truth(x_values)-truth(train[feature].to_numpy()).mean(), "k--", label="Centered truth")
        axis.plot(x_values, gam_effect, color=PINK, lw=2.4, label="GAM")
        axis.plot(x_values, nam_effect, color=BLUE, lw=2.2, label="NAM")
        axis.set_title(display_names[feature])
        axis.set_xlabel("Feature value")
        axis.set_ylabel("Centered contribution")
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0, 0].legend(frameon=False, ncols=2)
    figure.suptitle(
        "Same data and additive scale, different model representations",
        fontsize=14,
        y=1.035,
    )
    figure.text(
        0.5,
        0.985,
        f"Held-out R² — GAM: {gam_r2:.3f} · NAM: {nam_r2:.3f}",
        ha="center",
        va="top",
        color="#4b5563",
    )
    _save(figure, "gam_nam_all_features.png")


def generate_shape_constraint_plot() -> None:
    """Compare an ordinary smooth with a monotone-increasing smooth."""
    rng = np.random.default_rng(9)
    x = np.sort(rng.uniform(0.0, 10.0, 180))
    signal = 1.0 + 4.0 / (1.0 + np.exp(-(x - 5.0)))
    data = pd.DataFrame({"y": signal + rng.normal(0.0, 0.45, x.size), "x": x})

    ordinary = GAM(
        formula='y ~ s(x, bs="ps", k=12)',
        family="gaussian",
        optimize_smoothing=True,
        smoothing_method="gcv",
        smoothing_optimizer="bfgs",
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

    derivative = increasing.derivative(smooth_number=1, deriv=1)
    values = np.asarray(derivative.derivative).reshape(-1)
    uncertainty = np.asarray(derivative.se).reshape(-1)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(x, values, label="First derivative")
    axis.fill_between(x, values - 1.96 * uncertainty, values + 1.96 * uncertainty, alpha=.2)
    axis.axhline(0, color="black", linestyle="--")
    axis.set(xlabel="x", ylabel="Derivative of mean")
    axis.legend()
    _save(figure, "shape_constraint_derivative.png")


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


def _style_formula_axis(axis) -> None:
    """Apply the compact visual language used by the formula reference."""
    axis.grid(alpha=0.16)
    axis.spines[["top", "right"]].set_visible(False)


def generate_formula_reference_plots() -> None:
    """Fit the small examples used by the visual formula reference."""
    rng = np.random.default_rng(57)

    # Smooth continuous effect.
    x = np.sort(rng.uniform(0.0, 10.0, 240))
    y = 1.4 * np.sin(0.75 * x) + 0.13 * x + rng.normal(0.0, 0.34, x.size)
    continuous_data = pd.DataFrame({"y": y, "x": x})
    continuous = GAM(
        formula='y ~ s(x, bs="cr", k=10, sp=0.4)', family="gaussian"
    ).fit(continuous_data)
    continuous_grid = pd.DataFrame({"x": np.linspace(0.0, 10.0, 240)})
    continuous_fit, continuous_se = continuous.predict(
        continuous_grid, type="response", return_se=True
    )
    figure, axis = plt.subplots(figsize=(6.8, 3.7))
    axis.scatter(x, y, s=12, color="#9ca3af", alpha=0.4, label="observed")
    axis.fill_between(
        continuous_grid["x"],
        continuous_fit - 2.0 * continuous_se,
        continuous_fit + 2.0 * continuous_se,
        color=PINK_FILL,
        alpha=0.65,
        linewidth=0,
        label="approx. 95% interval",
    )
    axis.plot(continuous_grid["x"], continuous_fit, color=PINK, lw=2.5)
    axis.set(xlabel="x", ylabel="Expected response")
    axis.legend(frameon=False, ncols=2, fontsize=8)
    _style_formula_axis(axis)
    _save(figure, "formula_smooth_continuous.png")

    # Cyclic effect with observed endpoints so the fitted domain is exactly 0--24.
    hour = np.linspace(0.0, 24.0, 260)
    demand = (
        4.2 * np.sin(2.0 * np.pi * (hour - 6.0) / 24.0)
        + rng.normal(0.0, 0.45, hour.size)
    )
    periodic_data = pd.DataFrame({"demand": demand, "hour": hour})
    periodic = GAM(
        formula='demand ~ s(hour, bs="cc", k=12, sp=0.5)', family="gaussian"
    ).fit(periodic_data)
    periodic_grid = pd.DataFrame({"hour": np.linspace(0.0, 24.0, 300)})
    periodic_fit = periodic.predict(periodic_grid, type="response")
    figure, axis = plt.subplots(figsize=(6.8, 3.7))
    axis.scatter(hour, demand, s=11, color="#9ca3af", alpha=0.32)
    axis.plot(periodic_grid["hour"], periodic_fit, color=BLUE, lw=2.5)
    axis.scatter(
        [0.0, 24.0],
        periodic_fit[[0, -1]],
        s=50,
        color=PINK,
        edgecolor="white",
        linewidth=1,
        zorder=4,
        label="joined boundary",
    )
    axis.set(xticks=[0, 6, 12, 18, 24], xlabel="Hour", ylabel="Demand effect")
    axis.legend(frameon=False, fontsize=8)
    _style_formula_axis(axis)
    _save(figure, "formula_periodicity.png")

    # Group-specific smooth curves.
    n_group = 130
    segment = np.repeat(np.array(["A", "B", "C"], dtype=object), n_group)
    group_x = rng.uniform(-2.0, 2.0, segment.size)
    group_signal = np.select(
        [segment == "A", segment == "B", segment == "C"],
        [
            1.2 * np.sin(1.4 * group_x),
            0.8 * group_x**2 - 0.9,
            -1.15 * group_x,
        ],
    )
    group_data = pd.DataFrame(
        {
            "y": group_signal + rng.normal(0.0, 0.28, segment.size),
            "x": group_x,
            "segment": pd.Categorical(segment),
        }
    )
    grouped = GAM(
        formula=(
            'y ~ s(x, segment, bs="fs", k=7, xt="cr", '
            "sp=[0.6, 0.6, 0.6])"
        ),
        family="gaussian",
    ).fit(group_data)
    figure, axis = plt.subplots(figsize=(6.8, 3.7))
    palette = {"A": PINK, "B": BLUE, "C": GREEN}
    curve_x = np.linspace(-2.0, 2.0, 220)
    for label in ("A", "B", "C"):
        grid = pd.DataFrame(
            {
                "x": curve_x,
                "segment": pd.Categorical(
                    np.repeat(label, curve_x.size), categories=["A", "B", "C"]
                ),
            }
        )
        axis.plot(
            curve_x,
            grouped.predict(grid, type="response"),
            color=palette[label],
            lw=2.3,
            label=f"Segment {label}",
        )
    axis.set(xlabel="x", ylabel="Group-specific response")
    axis.legend(frameon=False, ncols=3, fontsize=8)
    _style_formula_axis(axis)
    _save(figure, "formula_group_curves.png")

    # Random intercepts, shown as fitted deviations from the grand mean.
    group_labels = np.array([f"G{i}" for i in range(1, 11)], dtype=object)
    group_shift = np.array([-1.25, -0.8, -0.45, -0.2, -0.05, 0.15, 0.3, 0.55, 0.9, 1.2])
    observed_group = rng.choice(group_labels, size=420, replace=True)
    shift_lookup = dict(zip(group_labels, group_shift, strict=True))
    random_data = pd.DataFrame(
        {
            "y": np.array([shift_lookup[value] for value in observed_group])
            + rng.normal(0.0, 0.72, observed_group.size),
            "group": pd.Categorical(observed_group, categories=group_labels),
        }
    )
    random_effect = GAM(
        formula='y ~ s(group, bs="re", sp=1.0)', family="gaussian"
    ).fit(random_data)
    random_grid = pd.DataFrame(
        {"group": pd.Categorical(group_labels, categories=group_labels)}
    )
    estimated_shift = random_effect.predict(random_grid, type="response")
    estimated_shift = estimated_shift - np.mean(estimated_shift)
    figure, axis = plt.subplots(figsize=(6.8, 3.7))
    axis.axhline(0.0, color="#9ca3af", lw=1)
    axis.vlines(group_labels, 0.0, estimated_shift, color=PINK_FILL, lw=3)
    axis.scatter(group_labels, estimated_shift, s=44, color=PINK, zorder=3)
    axis.set(xlabel="Group", ylabel="Estimated baseline deviation")
    _style_formula_axis(axis)
    _save(figure, "formula_random_effects.png")

    # Offset model: the fitted rate is shared, while exposure scales counts.
    offset_x = rng.uniform(0.0, 4.0, 520)
    exposure = rng.uniform(0.35, 2.4, offset_x.size)
    log_rate = 0.25 + 0.55 * np.sin(1.4 * offset_x) + 0.22 * offset_x
    count = rng.poisson(exposure * np.exp(log_rate))
    offset_data = pd.DataFrame(
        {
            "count": count,
            "x": offset_x,
            "exposure": exposure,
            "log_exposure": np.log(exposure),
        }
    )
    offset_model = GAM(
        formula=(
            'count ~ s(x, bs="cr", k=9, sp=0.5) '
            "+ offset(log_exposure)"
        ),
        family="poisson",
    ).fit(offset_data)
    figure, axis = plt.subplots(figsize=(6.8, 3.7))
    offset_grid_x = np.linspace(0.0, 4.0, 220)
    for value, color in zip((0.5, 1.0, 2.0), (GOLD, PINK, BLUE), strict=True):
        grid = pd.DataFrame(
            {
                "x": offset_grid_x,
                "exposure": value,
                "log_exposure": np.log(value),
            }
        )
        axis.plot(
            offset_grid_x,
            offset_model.predict(grid, type="response"),
            color=color,
            lw=2.2,
            label=f"Exposure = {value:g}",
        )
    axis.set(xlabel="x", ylabel="Expected count")
    axis.legend(frameon=False, fontsize=8)
    _style_formula_axis(axis)
    _save(figure, "formula_exposure_offset.png")


def generate_tensor_constructor_comparison() -> None:
    """Fit te, ti, and t2 representations to the same response surface."""
    rng = np.random.default_rng(73)
    n = 650
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.uniform(-1.0, 1.0, n)
    response = (
        0.9 * np.sin(np.pi * x)
        - 0.7 * z
        + 1.7 * x * z
        + 0.8 * np.exp(-7.0 * ((x - 0.35) ** 2 + (z + 0.25) ** 2))
        + rng.normal(0.0, 0.16, n)
    )
    data = pd.DataFrame({"response": response, "x": x, "z": z})
    formulas = {
        "te(): one full tensor term": (
            "response ~ te(x, z, bs=c('cr', 'cr'), k=c(7, 7))"
        ),
        "ti(): main effects + pure interaction": (
            "response ~ s(x, bs='cr', k=7) + s(z, bs='cr', k=7) "
            "+ ti(x, z, bs=c('cr', 'cr'), k=c(7, 7))"
        ),
        "t2(): ANOVA-style penalty blocks": (
            "response ~ t2(x, z, bs=c('cr', 'cr'), k=c(7, 7))"
        ),
    }
    models = {
        label: GAM(
            formula=formula,
            family="gaussian",
            optimize_smoothing=True,
            smoothing_method="reml",
        ).fit(data)
        for label, formula in formulas.items()
    }

    axis_values = np.linspace(-1.0, 1.0, 70)
    grid_x, grid_z = np.meshgrid(axis_values, axis_values)
    grid = pd.DataFrame({"x": grid_x.ravel(), "z": grid_z.ravel()})
    fitted = {
        label: model.predict(grid, type="response").reshape(grid_x.shape)
        for label, model in models.items()
    }
    color_limit = max(float(np.max(np.abs(values))) for values in fitted.values())

    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), sharex=True, sharey=True)
    image = None
    for axis, (label, values) in zip(axes, fitted.items(), strict=True):
        image = axis.contourf(
            grid_x,
            grid_z,
            values,
            levels=np.linspace(-color_limit, color_limit, 22),
            cmap="RdBu_r",
            extend="both",
        )
        axis.contour(
            grid_x,
            grid_z,
            values,
            levels=8,
            colors="#374151",
            linewidths=0.4,
            alpha=0.5,
        )
        axis.set_title(label, fontsize=10)
        axis.set_xlabel("x")
        axis.set_aspect("equal")
    axes[0].set_ylabel("z")
    figure.subplots_adjust(wspace=0.18, right=0.88)
    colorbar_axis = figure.add_axes([0.9, 0.17, 0.018, 0.66])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Fitted response")
    figure.savefig(
        DOCS_ROOT / "_static" / "formula_tensor_comparison.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


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
    generate_gam_nam_comparison()
    generate_shape_constraint_plot()
    generate_interaction_plot()
    generate_formula_reference_plots()
    generate_tensor_constructor_comparison()
    generate_neural_comparison()
    generate_distributional_plot()


if __name__ == "__main__":
    main()
