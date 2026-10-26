"""SCAM-specific display numerics from scam/R/plot.r and scam.check.R."""

from __future__ import annotations

import warnings

import numpy as np

from ..model_state import (
    _coef_full,
    _fit_state,
    _term_blocks_seq,
    _term_full_coefficient_indices,
)
from ..predict.predictions import _term_has_absorbed_constraint
from ..predict.terms import _prediction_term_groups
from ..term_labels import mgcv_term_display_label
from .plots import (
    _activate_by_column,
    _new_feature_matrix,
    _term_numeric_features,
    _training_lpmatrix,
    prepare_plot_gam_data,
)
from .vis import prepare_vis_gam_data


def _shape_term(term):
    return bool(np.any(getattr(term, "positive_coefficient_mask", False)))


def prepare_plot_scam_data(
    model,
    *,
    residuals=False,
    se=True,
    se_with_mean=False,
    unconditional=False,
    **kwargs,
):
    """Prepare plot.scam curves/surfaces and its distinct uncertainty/residuals.

    Shape covariance is already transported to SCAM's ``Vp.t`` coordinates by
    the fitted model. Grid centering below is the fitted-function equivalent of
    plot.r's QR ``XZa %*% RZaR`` projection, retaining the upstream distinction
    between its plotted values and its centered standard errors.
    """
    prepared = prepare_plot_gam_data(
        model,
        residuals=False,
        se=False,
        se_with_mean=False,
        unconditional=False,
        **kwargs,
    )
    blocks = [t for t in _term_blocks_seq(model) if t.term_type != "parametric"]
    beta = np.asarray(_coef_full(model), dtype=float)
    covariance = np.asarray(model.vcov(), dtype=float)
    unconditional_cov = covariance
    if unconditional:
        unconditional_cov = np.asarray(model.vcov(unconditional=True), dtype=float)
    residual_array = np.asarray(residuals)
    partial = bool(residual_array.size > 1 or residual_array.item())
    weighted_residuals = None
    if partial:
        if residual_array.size > 1 and residual_array.size == len(model.y_):
            weighted_residuals = np.asarray(residual_array, dtype=float).reshape(-1)
        else:
            if residual_array.size > 1:
                warnings.warn(
                    "residuals argument to plot.scam is wrong length: ignored",
                    stacklevel=2,
                )
            state = _fit_state(model)
            weights = getattr(state, "working_weights", None)
            if weights is None:
                raise NotImplementedError(
                    "SCAM partial residuals require fitted Newton weights"
                )
            root_weights = np.sqrt(np.asarray(weights, dtype=float))
            weighted_residuals = (
                np.asarray(model.residuals(type="working"))
                * root_weights
                / np.mean(root_weights)
            )
        term_values = np.asarray(model.predict(type="terms"), dtype=float)
        labels = [str(g["label"]) for g in _prediction_term_groups(model)]
    for term, plot in zip(blocks, prepared["pd"], strict=True):
        if not plot.get("plot_me"):
            continue
        shape = _shape_term(term)
        dimensions = _term_numeric_features(model, term)
        if shape and len(dimensions) not in {1, 2}:
            raise NotImplementedError(
                "SCAM displays support scalar univariate and bivariate smooths"
            )
        if plot.get("kind") not in {"1d", "2d"}:
            continue
        rows = (
            len(plot["x"]) if len(dimensions) == 1 else len(plot["x"]) * len(plot["y"])
        )
        new = _new_feature_matrix(model, rows)
        new[:, dimensions[0][0]] = (
            plot["x"] if len(dimensions) == 1 else np.tile(plot["x"], len(plot["y"]))
        )
        if len(dimensions) == 2:
            new[:, dimensions[1][0]] = np.repeat(plot["y"], len(plot["x"]))
        _activate_by_column(model, term, new)
        basis = np.asarray(term.predict_matrix(new), dtype=float)
        indices = _term_full_coefficient_indices(model, term)
        code = str(term.basis_name).lower()
        se_basis = basis
        if shape and (len(dimensions) == 2 or code in {"mpiby", "mpdby", "ipo", "dpo"}):
            se_basis = basis - basis.mean(axis=0)
        if shape and code == "dpo":
            # plot.scam drops the *last* QR direction for dpo, unlike ipo.
            # Its omitted final prediction column is the constant function.
            full_basis = np.column_stack([basis, np.ones(rows)])
            _, R = np.linalg.qr(full_basis, mode="reduced")
            Q_constraint, _ = np.linalg.qr(
                full_basis.sum(axis=0)[:, None], mode="complete"
            )
            retained = Q_constraint[:, :-1]
            R_reduced = R[:-1]
            projected = (
                full_basis @ retained @ np.linalg.solve(R_reduced @ retained, R_reduced)
            )
            se_basis = projected[:, :-1]
        fit_basis = se_basis if shape and len(dimensions) == 2 else basis
        fit = fit_basis @ beta[indices]
        exclude = np.asarray(
            plot.get("exclude", np.zeros(rows, dtype=bool)), dtype=bool
        )
        fit[exclude] = np.nan
        plot["fit"] = fit
        if bool(se):
            if se_with_mean and not shape and _term_has_absorbed_constraint(term):
                design = np.tile(np.mean(_training_lpmatrix(model), axis=0), (rows, 1))
                design[:, indices] = basis
                variance = np.sum((design @ unconditional_cov) * design, axis=1)
            else:
                cov = covariance if shape else unconditional_cov
                variance = np.sum(
                    (se_basis @ cov[np.ix_(indices, indices)]) * se_basis, axis=1
                )
            standard_error = np.sqrt(np.maximum(variance, 0))
            # plot.scam masks fitted values but leaves SCOP SE outside its grid.
            if not shape:
                standard_error[exclude] = np.nan
            multiplier = (
                (2.0 if len(dimensions) == 1 else 1.0)
                if isinstance(se, (bool, np.bool_))
                else max(0.0, float(se))
            )
            width = standard_error * multiplier
            plot.update(
                se_fit=standard_error,
                se=width,
                se_mult=np.array([multiplier]),
                ll=(fit - width)[:, None],
                ul=(fit + width)[:, None],
                plot_ci=True,
            )
        if partial:
            label = mgcv_term_display_label(term)
            if label in labels:
                plot["p_resid"] = (
                    term_values[:, labels.index(label)] + weighted_residuals
                )
    prepared["partial_resids"] = partial
    if kwargs.get("scale", -1) == -1 and kwargs.get("ylim") is None:
        ranges = []
        for plot in prepared["pd"]:
            if not plot.get("plot_me") or not plot.get("scale"):
                continue
            lower = plot["ll"][:, 0] if plot.get("plot_ci") else plot["fit"]
            upper = plot["ul"][:, 0] if plot.get("plot_ci") else plot["fit"]
            ranges.extend([np.nanmin(lower), np.nanmax(upper)])
            if "p_resid" in plot:
                ranges.extend([np.nanmin(plot["p_resid"]), np.nanmax(plot["p_resid"])])
        if ranges:
            transform = prepared["trans"]
            shift = prepared["shift"]
            prepared["ylim"] = (
                float(transform(min(ranges) + shift)),
                float(transform(max(ranges) + shift)),
            )
    return prepared


def prepare_vis_scam_data(model, **kwargs):
    """vis.scam uses the same conditioning grid as vis.gam and SCAM prediction."""
    return prepare_vis_gam_data(model, **kwargs)


def prepare_qq_scam_data(
    model,
    *,
    rep=0,
    level=0.9,
    s_rep=10,
    type="deviance",
    seed=None,
    quantile_permutations=None,
    response_draws=None,
):
    """qq.scam shares qq.gam's simulation/quantile algorithm and residuals."""
    from .qq import prepare_qq_gam_data

    return prepare_qq_gam_data(
        model,
        rep=rep,
        level=level,
        s_rep=s_rep,
        type=type,
        seed=seed,
        quantile_permutations=quantile_permutations,
        response_draws=response_draws,
    )


def prepare_scam_check_data(
    model,
    *,
    type="deviance",
    old_style=False,
    rep=0,
    level=0.9,
    s_rep=10,
    seed=None,
    setseed=None,
    quantile_permutations=None,
    response_draws=None,
):
    """scam.check diagnostic panels; unlike gam.check it has no k-index test."""
    from ..model_state import _fitted_eta, _fitted_mu
    from .check_plots import _r_histogram, prepare_gam_check_data
    from .qq import _normal_qq

    if type != "rquantile":
        result = prepare_gam_check_data(
            model,
            type=type,
            old_style=old_style,
            rep=rep,
            level=level,
            s_rep=s_rep,
            seed=seed,
            quantile_permutations=quantile_permutations,
            response_draws=response_draws,
        )
    else:
        residuals = np.asarray(
            model.residuals(
                type="rquantile", setseed=setseed if setseed is not None else seed
            )
        )
        result = {
            "qq": _normal_qq(residuals, type),
            "residual_vs_linear": {
                "x": np.asarray(_fitted_eta(model)),
                "y": residuals,
                "xlab": "linear predictor",
                "ylab": "rquantile residuals",
                "title": "Resids vs. linear pred.",
            },
            "histogram": _r_histogram(residuals),
            "response_vs_fitted": {
                "x": np.asarray(_fitted_mu(model)),
                "y": np.asarray(model.y_),
                "xlab": "fitted values",
                "ylab": "response",
                "title": "Response vs. Fitted Values",
            },
            "type": type,
            "old_style": bool(old_style),
        }
    result["histogram"]["xlab"] = f"{type} residuals"
    result["backend"] = "scam"
    return result
