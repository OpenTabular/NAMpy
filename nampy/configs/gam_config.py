# configs/gam_config.py
from dataclasses import dataclass
from typing import Optional, Sequence, Union

from ..gam.specs import TermSpec


@dataclass
class DefaultGAMConfig:
    # spline construction
    k: int = 10
    basis: str = (
        "tp"  # mgcv-compatible default for bare s(...); main effects also support cr, cs, cc, ps, ts
    )
    select: bool = False
    main_effects: bool = True
    tensor_terms: Optional[Sequence[TermSpec]] = None
    knots: Optional[Union[dict, list, tuple]] = None
    min_sp: Optional[float] = None
    drop_intercept: Optional[bool] = None

    # smoothing
    smoothing_params: Optional[Union[float, Sequence[float]]] = None
    optimize_smoothing: bool = False
    smoothing_method: str = "fixed"  # fixed, gcv, ml, reml, laml
    smoothing_optimizer: str = "lbfgsb"  # lbfgsb, outer_newton
    sp_log_bounds: tuple[float, float] = (-80.0, 20.0)
    score_gamma: float = 1.0

    # model structure
    fit_intercept: bool = True

    # PIRLS controls
    max_irls_iter: int = 100
    # Exact mgcv parity for non-Gaussian fixed-sp and outer-loop evaluations
    # requires a tighter PIRLS stop than the original loose default.
    irls_tol: float = 1e-11
    max_step_halving: int = 25

    # inference / covariance
    covariance: str = "bayes"  # bayes, freq, kass_steffey, wood
    compute_edf: bool = True

    # diagnostics
    enable_k_check: bool = False
    enable_concurvity: bool = False
