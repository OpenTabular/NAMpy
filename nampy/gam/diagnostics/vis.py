"""Backend-neutral prediction surfaces from mgcv R/plots.r::vis.gam."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..formula.parse import all_vars1
from ..model_state import _require_fitted
from .plots import exclude_too_far


def _raw_predictors(model):
    frame = getattr(model, "vis_training_data_", None)
    if frame is None:
        frame = pd.DataFrame(model.X_, columns=model.feature_names)
    names = []
    parsed = getattr(model, "formula_", None)
    if parsed is not None:
        for component in parsed.components:
            for expression in component.fake_names:
                for name in all_vars1(expression):
                    if name in frame and name not in names:
                        names.append(name)
    # Dot expansion and array-fitting have no useful source expression list.
    names.extend(name for name in frame if name not in names)
    return frame.loc[:, names]


def _summary(series):
    """mgcv::variable.summary: observed type-3 median, or modal factor."""
    if isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(
        series.dtype
    ):
        raw = series.dropna()
        if isinstance(series.dtype, pd.CategoricalDtype):
            levels = list(series.cat.categories)
        elif all(isinstance(value, str) for value in raw):
            levels = sorted(raw.unique().tolist())
        else:
            levels = None
        if levels is not None:
            counts = np.array([(raw == value).sum() for value in levels])
            return {
                "kind": "factor",
                "levels": levels,
                "default": levels[int(np.argmax(counts))],
            }
    values = series.to_numpy()
    if values.dtype == object and any(np.asarray(v).ndim > 0 for v in values):
        matrix = np.stack([np.asarray(v, dtype=float) for v in values])
        return {
            "kind": "matrix",
            "default": np.quantile(matrix, 0.5, axis=0, method="closest_observation"),
            "width": matrix.shape[1],
        }
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("View predictors must contain finite observations.")
    return {
        "kind": "numeric",
        "range": (float(values.min()), float(values.max())),
        "default": float(np.quantile(values, 0.5, method="closest_observation")),
    }


def _factor_sequence(levels, n_grid):
    """vis.gam's fac.seq, including final-level remainder and truncation."""
    if len(levels) > n_grid:
        values = list(levels[:n_grid])
        # factor(levels(fac)) recreates alphabetically sorted levels here.
        plotting_levels = sorted(levels)
    else:
        count = n_grid // len(levels)
        values = [value for value in levels for _ in range(count)]
        values += [levels[-1]] * (n_grid - len(values))
        plotting_levels = levels
    codes = np.array(
        [plotting_levels.index(value) + 1 for value in values], dtype=float
    )
    return np.asarray(values, dtype=object), codes


def prepare_vis_gam_data(
    model,
    *,
    view=None,
    cond=None,
    n_grid=30,
    too_far=0,
    type="link",
    se=-1,
    lp=1,
    zlim=None,
):
    """Prepare a full-model surface, conditioned like mgcv's ``vis.gam``.

    ``fit``, ``se_fit`` and ``excluded`` are flat with the first view varying
    fastest. Reshape with C order for plotting rows along y, columns along x.
    ``x``/``y`` are plotting axes; ``grid_values`` retain original factor labels.
    """
    _require_fitted(model)
    if not np.isfinite(n_grid) or int(n_grid) != n_grid or n_grid < 2:
        raise ValueError("n_grid must be an integer >= 2.")
    n_grid = int(n_grid)
    if type not in {"link", "response"}:
        raise ValueError('type must be "link" or "response"')
    if not np.isfinite(se) or not np.isfinite(lp) or not np.isfinite(too_far):
        raise ValueError("se, lp and too_far must be finite.")
    frame = _raw_predictors(model)
    summaries = {name: _summary(frame[name]) for name in frame}

    def varies(summary):
        return (
            len(summary["levels"]) > 1
            if summary["kind"] == "factor"
            else summary["range"][0] != summary["range"][1]
            if summary["kind"] == "numeric"
            else False
        )

    if view is None:
        view = [name for name, summary in summaries.items() if varies(summary)][:2]
        if len(view) < 2:
            raise ValueError(
                "Model does not seem to have enough terms to do anything useful"
            )
    else:
        view = [view] if isinstance(view, str) else list(view)
        if len(view) != 2 or any(name not in summaries for name in view):
            raise ValueError(f"view variables must be two of {list(summaries)}")
        if any(summaries[name]["kind"] == "matrix" for name in view):
            raise ValueError(
                "View variables must be simple numeric or factor variables."
            )
        if any(not varies(summaries[name]) for name in view):
            raise ValueError("View variables must contain more than one value.")
    cond = {} if cond is None else dict(cond)
    n_rows = n_grid * n_grid
    newdata = pd.DataFrame(index=np.arange(n_rows))
    for name, summary in summaries.items():
        value = cond.get(name, summary["default"])
        if value is None:
            value = summary["default"]
        if summary["kind"] == "matrix":
            row = np.resize(
                np.asarray(value, dtype=float).reshape(-1), summary["width"]
            )
            newdata[name] = [row.copy() for _ in range(n_rows)]
        else:
            if np.asarray(value).size != 1:
                raise ValueError(f"Condition for {name!r} must be a single value.")
            newdata[name] = np.repeat(np.asarray(value).reshape(-1)[0], n_rows)
    axes, grid_values, codes = [], [], []
    factor_levels = {}
    for name in view:
        summary = summaries[name]
        if summary["kind"] == "factor":
            values, code = _factor_sequence(summary["levels"], n_grid)
            observed = set(frame[name].dropna())
            if any(value not in observed for value in values):
                raise ValueError(
                    f"View {name!r} includes unobserved factor levels; "
                    "remove unused categories before fitting."
                )
            factor_levels[name] = list(summary["levels"])
            axes.append(code.copy())
            codes.append(code)
        else:
            values = np.linspace(*summary["range"], n_grid)
            axes.append(values.copy())
            codes.append(None)
        grid_values.append(values)
    newdata[view[0]] = np.tile(grid_values[0], n_grid)
    newdata[view[1]] = np.repeat(grid_values[1], n_grid)
    fit, standard_error = model.predict(newdata, return_se=True, type=type)
    fit, standard_error = (
        np.asarray(fit, dtype=float),
        np.asarray(standard_error, dtype=float),
    )
    selected_lp = 1
    if fit.ndim == 2:
        selected_lp = int(np.clip(np.rint(lp), 1, fit.shape[1]))
        fit, standard_error = (
            fit[:, selected_lp - 1],
            standard_error[:, selected_lp - 1],
        )
    fit, standard_error = fit.copy(), standard_error.copy()
    unmasked_fit = fit.copy()
    excluded = np.zeros(n_rows, dtype=bool)
    if too_far > 0:
        if factor_levels:
            raise ValueError(
                "too_far requires numeric view variables, as in mgcv::exclude.too.far."
            )
        excluded = exclude_too_far(
            newdata[view[0]],
            newdata[view[1]],
            frame[view[0]],
            frame[view[1]],
            dist=too_far,
        )
        fit[excluded] = standard_error[excluded] = np.nan
    if codes[0] is not None:
        axes[0] = np.linspace(codes[0].min() - 0.5, codes[0].max() + 0.5, n_grid)
    if codes[1] is not None:
        # Preserve plots.r's use of min(m1), even for the second factor axis.
        axes[1] = np.linspace(axes[0].min() - 0.5, codes[1].max() + 0.5, n_grid)
    lower = fit - se * standard_error if se > 0 else None
    upper = fit + se * standard_error if se > 0 else None
    if zlim is not None:
        limits = np.asarray(zlim, dtype=float).reshape(-1)
        if limits.size != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError("Something wrong with zlim")
    else:
        lo, hi = (lower, upper) if se > 0 else (fit, fit)
        if not np.isfinite(lo).any() or not np.isfinite(hi).any():
            raise ValueError("No finite surface values remain to determine zlim.")
        limits = np.array([np.nanmin(lo), np.nanmax(hi)])
        if se <= 0 and limits[0] == limits[1]:
            limits += [-1, 1]
    return {
        "view": view,
        "x": axes[0],
        "y": axes[1],
        "fit": fit,
        "se_fit": standard_error,
        "lower": lower,
        "upper": upper,
        "newdata": newdata,
        "grid_values": grid_values,
        "factor_levels": factor_levels,
        "excluded": excluded,
        "unmasked_fit": unmasked_fit,
        "n_grid": n_grid,
        "se": float(se),
        "lp": selected_lp,
        "type": type,
        "zlim": limits,
        "zlab": "linear predictor" if type == "link" else "response",
    }
