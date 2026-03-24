import numpy as np

from .backends import solve_fit
from .offsets import coerce_offset_array
from .state import assign_fit_solution


def fit_model_core(model, X, feature_names, y, offset=None, optimize_smoothing=None, smoothing_method=None):
    """
    Orchestration-only fit entry point.

    This keeps the public/core class stable while moving the actual fit
    mechanics into gam.fit.
    """
    X = model._coerce_feature_matrix(X)
    y = model.family.validate_y(y)
    offset = coerce_offset_array(offset, X.shape[0])

    model.X_ = X
    model.feature_names = list(feature_names)
    model.y_ = y
    model.offset_train_ = offset
    model.n_samples_ = X.shape[0]

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
