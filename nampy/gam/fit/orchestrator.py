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

import numpy as np

from .backends import solve_fit
from .offsets import coerce_offset_array
from .state import assign_fit_solution


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

    sol = solve_fit(model, y, model.smoothing_params, weights=model.prior_weights_)

    assign_fit_solution(model, sol)
    if method != "fixed" and not (getattr(model, "_optim_trace", None) or []):
        inner_trace = list(getattr(sol, "inner_trace", None) or [])
        if inner_trace:
            fixed_mask = (
                np.zeros(model.n_smoothing_params_, dtype=bool)
                if model.smoothing_fixed_mask_ is None
                else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
            )
            free_vals = np.asarray(model.smoothing_params[~fixed_mask], dtype=np.float64)
            log_sp = (
                np.log(np.maximum(free_vals, 1e-300))
                if free_vals.size > 0
                else np.empty((0,), dtype=np.float64)
            )
            model._optim_trace = [
                {
                    **dict(row),
                    "iter": int(row.get("iter", i + 1)),
                    "log_sp": log_sp.copy(),
                    "criterion": float(
                        row.get(
                            "penalized_deviance_conv",
                            row.get("penalized_deviance", np.nan),
                        )
                    ),
                    "gradient": np.asarray(
                        [float(row["grad_inf_norm"])], dtype=np.float64
                    )
                    if row.get("grad_inf_norm", None) is not None
                    else None,
                    "rank_info": {
                        "pirls_inner": True,
                        "converged_here": bool(row.get("converged_here", False)),
                    },
                }
                for i, row in enumerate(inner_trace)
            ]

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
