"""
Main fitting orchestration: compile designs, optimise smoothing, solve, store.

:func:`fit_model_core` is the single entry point called by GAM model classes.
It sequences the full fit:

1. Coerce and store input data on the model.
2. Compile the term designs and penalty blocks.
3. Optionally optimise smoothing parameters (GCV / REML / ML).
4. Solve the penalized system at the chosen smoothing parameters.
5. Assign the fitted coefficients and diagnostics to the model.

Everything below this layer is stateless with respect to the model object.
"""

from dataclasses import replace

import numpy as np

from .backends import solve_fit
from .offsets import coerce_offset_array
from .state import assign_fit_solution
from ..smoothness.criteria.gaussian_dyn import criterion_ml_reml_gaussian_dynamic_joint
from ..smoothness.criteria.ml_reml import resolve_ml_reml_scoring_backend
from ..smoothness.criteria.penalty import _static_penalty_null_dim
from ..smoothness.criteria.gaussian_reml_algebra import (
    gaussian_weighted_residual_sum_squares,
    prior_weights_diagonal_from_fit,
    quadratic_form_penalty,
)


def fit_model_core(
    model,
    X,
    feature_names,
    y,
    offset=None,
    optimize_smoothing=None,
    smoothing_method=None,
    sample_weight=None,
):
    """
    Compile, (optionally) optimise, solve, and store — full GAM fit in one call.

    Parameters
    ----------
    model
        GAM model object.  Must implement ``_coerce_feature_matrix``,
        ``_compile_designs``, ``optimize_smoothing_params``, and
        ``_build_fit_result``.
    X
        Raw feature matrix, shape ``(n, p)``.
    feature_names
        Ordered list of feature names corresponding to ``X`` columns.
    y
        Response vector, shape ``(n,)``.
    offset
        Optional fixed offset on the link scale, shape ``(n,)``.
    optimize_smoothing
        Override ``model.optimize_smoothing``.  If True and ``smoothing_method``
        is not ``"fixed"``, outer smoothing-parameter optimisation runs before
        the final solve.
    smoothing_method
        Override ``model.smoothing_method`` (e.g. ``"reml"``, ``"gcv"``).
    sample_weight
        Optional per-observation prior weights, shape ``(n,)``.
    """
    X = model._coerce_feature_matrix(X)
    y = model.family.validate_y(y)
    offset = coerce_offset_array(offset, X.shape[0])

    model.X_ = X
    model.feature_names = list(feature_names)
    model.y_ = y
    model.offset_train_ = offset
    model.n_samples_ = X.shape[0]
    model.prior_weights_ = None
    if sample_weight is not None:
        sw = np.asarray(sample_weight, dtype=np.float64).ravel()
        if sw.shape[0] != int(model.n_samples_):
            raise ValueError(
                "sample_weight must have the same length as y "
                f"({model.n_samples_}), got {sw.shape[0]}."
            )
        if not np.all(np.isfinite(sw)) or np.any(sw < 0.0):
            raise ValueError("sample_weight must be finite and non-negative.")
        if float(np.sum(sw)) <= 0.0:
            raise ValueError("sample_weight must sum to a positive value.")
        model.prior_weights_ = sw.copy()

    model._compile_designs(X, model.feature_names)

    if model._has_tensor_terms() and model.family.name != "gaussian":
        raise NotImplementedError(
            "Tensor-product smooths are enabled only for Gaussian families in this phase."
        )

    if optimize_smoothing is None:
        optimize_smoothing = model.optimize_smoothing
    method = model._resolve_smoothing_method(
        model.smoothing_method if smoothing_method is None else smoothing_method
    )

    if optimize_smoothing and method != "fixed":
        if not model._supports_smoothing_method(method):
            if method in {"ml", "reml", "laml"}:
                model._raise_ml_reml_backend_error(method)
            raise NotImplementedError(
                f"Automatic smoothing selection with method={method!r} is not "
                f"supported for family={model.family.name!r}."
            )
        user_initial_sp = None
        if hasattr(model, "hparams") and hasattr(model.hparams, "get"):
            user_initial_sp = model.hparams.get("smoothing_params", None)
        model.optimize_smoothing_params(
            y=y,
            initial_smoothing_params=(
                model.smoothing_params if user_initial_sp is not None else None
            ),
            method=method,
            optimizer=model.smoothing_optimizer,
        )
    else:
        model._optim_method = "fixed"
        model._optim_result = None
        model._optim_trace = None
        model._optim_used_gradient = False
        model._optim_used_hessian = False
        model.smoothing_score_ = None

    sol = solve_fit(model, y, model.smoothing_params)

    assign_fit_solution(model, sol)

    # For Gaussian REML/LAML with the exact (profiled Laplace) backend, the optimizer
    # stores the profiled mixed-model criterion value, which differs from mgcv's reported
    # gcv.ubre by a large constant.  Recompute using the Wood-style joint formula at the
    # profiled sigma^2 so that criterion_value matches mgcv output.
    _optim_m = getattr(model, "_optim_method", None)
    if (
        _optim_m in {"reml", "laml"}
        and getattr(model, "_gaussian_reml_sigma2_opt_", None) is None
        and str(getattr(model.family, "name", "")).lower() == "gaussian"
    ):
        try:
            _backend = resolve_ml_reml_scoring_backend(model, method=_optim_m)
            if _backend == "gaussian_exact":
                _fixed_mask = (
                    np.zeros(model.n_smoothing_params_, dtype=bool)
                    if model.smoothing_fixed_mask_ is None
                    else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
                )
                _free_vals = np.asarray(model.smoothing_params[~_fixed_mask], dtype=np.float64)
                _log_free = (
                    np.log(np.maximum(_free_vals, 1e-300))
                    if _free_vals.size > 0
                    else np.empty((0,), dtype=np.float64)
                )
                _n_s = int(model.n_samples_)
                _w = prior_weights_diagonal_from_fit(sol, _n_s)
                _yv = np.asarray(y, dtype=np.float64).ravel()
                _mu_v = np.asarray(sol.mu, dtype=np.float64).ravel()
                _dev = gaussian_weighted_residual_sum_squares(_yv, _mu_v, _w)
                _P_pen = (
                    float(sol.penalty_quadratic)
                    if sol.penalty_quadratic is not None
                    else quadratic_form_penalty(
                        np.asarray(sol.coef_full, dtype=np.float64),
                        np.asarray(sol.penalty_matrix, dtype=np.float64),
                    )
                )
                _Mp = float(
                    _static_penalty_null_dim(model)
                    + int(bool(getattr(model, "fit_intercept", False)))
                )
                _nu = float(_n_s) - _Mp
                if np.isfinite(_nu) and _nu > 0.0:
                    _sigma2_prof = (_dev + _P_pen) / _nu
                    if _sigma2_prof > 0.0:
                        _log_s2 = float(np.log(_sigma2_prof))
                        _branch_m = "LAML" if _optim_m == "laml" else "REML"
                        _wood = float(
                            criterion_ml_reml_gaussian_dynamic_joint(
                                model, y, _log_free, _log_s2, method=_branch_m
                            )
                        )
                        if np.isfinite(_wood):
                            model.smoothing_score_ = _wood
        except Exception:
            pass

    reml_s2 = getattr(model, "_gaussian_reml_sigma2_opt_", None)
    if reml_s2 is not None and np.isfinite(reml_s2) and float(reml_s2) > 0.0:
        model.scale_ = float(reml_s2)
        if getattr(model, "fit_state_", None) is not None:
            model.fit_state_ = replace(model.fit_state_, scale=float(reml_s2))

    if model.smoothing_score_ is None and model._optim_method not in {None, "fixed"}:
        fixed_mask = (
            np.zeros(model.n_smoothing_params_, dtype=bool)
            if model.smoothing_fixed_mask_ is None
            else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
        )
        free_vals = np.asarray(model.smoothing_params[~fixed_mask], dtype=np.float64)
        log_free = (
            np.log(free_vals)
            if free_vals.size > 0
            else np.empty((0,), dtype=np.float64)
        )
        model.smoothing_score_ = float(
            model._criterion(y, log_free, method=model._optim_method)
        )

    model._fitted = True
    model.result_ = model._build_fit_result()
    return model
