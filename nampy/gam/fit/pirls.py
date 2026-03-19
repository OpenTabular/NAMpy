from .pirls_core import fit_pirls_core
from .state import FitCoreSolution


def solve_pirls_gam(
    Z,
    y,
    penalty_blocks,
    smoothing_params,
    family,
    fit_intercept=True,
    max_iter=100,
    tol=1e-8,
    max_step_halving=25,
    offset=None,
):
    return fit_pirls_core(
        Z=Z,
        y=y,
        penalty_blocks=penalty_blocks,
        smoothing_params=smoothing_params,
        family=family,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tol,
        max_step_halving=max_step_halving,
        offset=offset,
    )


def solve_pirls_fit(model, y, smoothing_params):
    """
    Penalized IRLS solve for one-predictor non-Gaussian GAMs.
    """
    sol = solve_pirls_gam(
        Z=model.Z,
        y=y,
        penalty_blocks=model.penalty_blocks_,
        smoothing_params=smoothing_params,
        family=model.family,
        fit_intercept=model.fit_intercept,
        max_iter=int(getattr(model, "max_irls_iter", 100)),
        tol=float(getattr(model, "irls_tol", 1e-8)),
        max_step_halving=int(getattr(model, "max_step_halving", 25)),
        offset=model.offset_train_,
    )
    return FitCoreSolution.from_dict(sol)
