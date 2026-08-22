from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace as _optimization_lifecycle_replace
from typing import Any, Callable

import pandas as pd

from tests.mgcv_parity_utils import (
    _make_binomial_data,
    _make_gamma_data,
    _make_gaussian_data,
    _make_gaussian_link_data,
    _make_negbin_data,
    _make_poisson_data,
)
from tests.mgcv_parity_utils import (
    _make_random_effect_data_noisy as _coverage_make_random_effect_data_noisy,
)


@dataclass(frozen=True)
class OptimizationLifecycleCase:
    case_id: str
    formula: str | list[str]
    family: str | dict[str, Any]
    mgcv_family: str
    method: str
    optimizer: str
    data_factory: Callable[..., pd.DataFrame]
    smoothing_optimizer: str | None = None
    select: bool = False
    weights_column: str | None = None
    compare_hat: bool = True
    exact_outer_info_trace: bool = True
    skip_coef_comparison: bool = False
    trace_atol: float = 1e-6
    trace_sp_atol: float | None = None
    cov_rtol: float = 3e-5
    cov_atol: float = 5e-8
    scalar_atol: float = 2e-4
    gam_kwargs: dict[str, Any] = field(default_factory=dict)
    status: str = "stable"
    known_gap_reason: str | None = None
    # Smoothing-parameter index groups whose assignment order is
    # mgcv-indeterminate (e.g. fs null-space penalties: upstream assigns one
    # sp per nat.param null column, and R's eigen orders those numerically-zero
    # eigenvalues by roundoff — mgcv itself flips the order under row
    # permutation of the same data; see the retained local stability probe).
    # The harness canonicalizes each side independently by descending final
    # log-sp inside each group before the strict comparison.
    exchangeable_sp_groups: tuple[tuple[int, ...], ...] = ()
    # Parallel to exchangeable_sp_groups: for each sp in a group, the public
    # coefficient columns tied to that sp, so coefficient-indexed surfaces
    # (covariance diagonals) can be permuted consistently.
    exchangeable_sp_coef_cols: tuple[tuple[tuple[int, ...], ...], ...] = ()
    # Set False when the unconditional covariance inherits more than
    # tolerance-level spread from an mgcv-internal indeterminacy (evidence
    # required in the case comment); Vp/Ve stay strict.
    compare_unconditional: bool = True


OPTIMIZATION_LIFECYCLE_CASES: list[OptimizationLifecycleCase] = [
    OptimizationLifecycleCase(
        case_id="poisson_reml_newton_two_cr",
        formula='y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        family="poisson",
        mgcv_family="poisson",
        method="REML",
        optimizer="newton",
        smoothing_optimizer="outer_newton",
        data_factory=_make_poisson_data,
        trace_atol=5e-7,
    ),
    OptimizationLifecycleCase(
        case_id="poisson_reml_bfgs_two_cr",
        formula='y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        family="poisson",
        mgcv_family="poisson",
        method="REML",
        optimizer="bfgs",
        smoothing_optimizer="bfgs",
        data_factory=_make_poisson_data,
        trace_atol=2e-5,
        trace_sp_atol=1e-5,
    ),
    OptimizationLifecycleCase(
        case_id="poisson_reml_efs_two_cr",
        formula='y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        family="poisson",
        mgcv_family="poisson",
        method="REML",
        optimizer="efs",
        smoothing_optimizer="efs",
        data_factory=_make_poisson_data,
        trace_atol=2e-5,
    ),
    OptimizationLifecycleCase(
        case_id="poisson_reml_optim_two_cr",
        formula='y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        family="poisson",
        mgcv_family="poisson",
        method="REML",
        optimizer="optim",
        smoothing_optimizer="optim",
        data_factory=_make_poisson_data,
        trace_atol=2e-5,
        gam_kwargs={"sp_log_bounds": (-80.0, 25.0)},
    ),
    OptimizationLifecycleCase(
        case_id="gaussian_reml_newton_two_cr",
        formula='y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        family="gaussian",
        mgcv_family="gaussian",
        method="REML",
        optimizer="newton",
        smoothing_optimizer="outer_newton",
        data_factory=_make_gaussian_data,
        trace_atol=5e-7,
    ),
    OptimizationLifecycleCase(
        case_id="binomial_reml_newton_cr",
        formula='y ~ s(x0, bs="cr", k=8)',
        family="binomial",
        mgcv_family="binomial",
        method="REML",
        optimizer="newton",
        smoothing_optimizer="outer_newton",
        data_factory=_make_binomial_data,
        skip_coef_comparison=True,
        trace_atol=2e-5,
        cov_rtol=5e-5,
        cov_atol=5e-7,
        scalar_atol=5e-4,
    ),
    OptimizationLifecycleCase(
        case_id="poisson_reml_newton_tensor_cr",
        formula='y ~ te(x0, x1, bs=["cr", "cr"], k=[6, 6])',
        family="poisson",
        mgcv_family="poisson",
        method="REML",
        optimizer="newton",
        smoothing_optimizer="outer_newton",
        data_factory=_make_poisson_data,
        skip_coef_comparison=True,
        trace_atol=2e-5,
        cov_rtol=7e-5,
        cov_atol=7e-7,
        scalar_atol=7e-4,
    ),
    OptimizationLifecycleCase(
        case_id="gamma_reml_newton_joint_scale_cr",
        formula='y ~ s(x0, bs="cr", k=8)',
        family="gamma",
        mgcv_family="gamma",
        method="REML",
        optimizer="newton",
        smoothing_optimizer="outer_newton",
        data_factory=_make_gamma_data,
        trace_atol=5e-7,
    ),
    OptimizationLifecycleCase(
        case_id="gamma_ml_newton_joint_scale_cr",
        formula='y ~ s(x0, bs="cr", k=8)',
        family="gamma",
        mgcv_family="gamma",
        method="ML",
        optimizer="newton",
        smoothing_optimizer="outer_newton",
        data_factory=_make_gamma_data,
        trace_atol=5e-7,
    ),
    OptimizationLifecycleCase(
        case_id="gamma_ml_bfgs_joint_scale_cr",
        formula='y ~ s(x0, bs="cr", k=8)',
        family="gamma",
        mgcv_family="gamma",
        method="ML",
        optimizer="bfgs",
        smoothing_optimizer="bfgs",
        data_factory=_make_gamma_data,
        trace_atol=5e-7,
        trace_sp_atol=5e-7,
        cov_rtol=1e-4,
        cov_atol=1e-6,
        scalar_atol=1e-3,
    ),
    OptimizationLifecycleCase(
        case_id="gamma_ml_optim_joint_scale_cr",
        formula='y ~ s(x0, bs="cr", k=8)',
        family="gamma",
        mgcv_family="gamma",
        method="ML",
        optimizer="optim",
        smoothing_optimizer="optim",
        data_factory=_make_gamma_data,
        trace_atol=3e-5,
        trace_sp_atol=3e-5,
        cov_rtol=1e-4,
        cov_atol=1e-6,
        scalar_atol=1e-3,
    ),
    OptimizationLifecycleCase(
        case_id="gaussian_log_reml_newton_joint_scale_cr",
        formula='y ~ s(x0, bs="cr", k=8)',
        family={"name": "gaussian", "link": "log"},
        mgcv_family="gaussian:log",
        method="REML",
        optimizer="newton",
        smoothing_optimizer="outer_newton",
        data_factory=_make_gaussian_link_data,
        trace_atol=5e-7,
    ),
    OptimizationLifecycleCase(
        case_id="gaussian_log_ml_newton_joint_scale_cr",
        formula='y ~ s(x0, bs="cr", k=8)',
        family={"name": "gaussian", "link": "log"},
        mgcv_family="gaussian:log",
        method="ML",
        optimizer="newton",
        smoothing_optimizer="outer_newton",
        data_factory=_make_gaussian_link_data,
        trace_atol=5e-7,
    ),
    OptimizationLifecycleCase(
        case_id="gaussian_log_reml_bfgs_joint_scale_cr",
        formula='y ~ s(x0, bs="cr", k=8)',
        family={"name": "gaussian", "link": "log"},
        mgcv_family="gaussian:log",
        method="REML",
        optimizer="bfgs",
        smoothing_optimizer="bfgs",
        data_factory=_make_gaussian_link_data,
        trace_atol=3e-5,
        trace_sp_atol=3e-5,
        cov_rtol=1e-4,
        cov_atol=1e-6,
        scalar_atol=1e-3,
    ),
    OptimizationLifecycleCase(
        case_id="gaussian_inverse_reml_newton_joint_scale_cr",
        formula='y ~ s(x0, bs="cr", k=8)',
        family={"name": "gaussian", "link": "inverse"},
        mgcv_family="gaussian:inverse",
        method="REML",
        optimizer="newton",
        smoothing_optimizer="outer_newton",
        data_factory=_make_gaussian_link_data,
        trace_atol=5e-7,
    ),
    OptimizationLifecycleCase(
        case_id="negbin_est_reml_newton_joint_theta_cr",
        formula='y ~ s(x0, bs="cr", k=8)',
        family={"name": "negbin", "theta": 1.8, "estimate_theta": True},
        mgcv_family="negbin_est:1.8",
        method="REML",
        optimizer="newton",
        smoothing_optimizer="outer_newton",
        data_factory=_make_negbin_data,
        trace_atol=2e-5,
    ),
    OptimizationLifecycleCase(
        case_id="negbin_est_reml_newton_joint_theta_weighted_cr",
        formula='y ~ s(x0, bs="cr", k=8)',
        family={"name": "negbin", "theta": 1.8, "estimate_theta": True},
        mgcv_family="negbin_est:1.8",
        method="REML",
        optimizer="newton",
        smoothing_optimizer="outer_newton",
        data_factory=_make_negbin_data,
        weights_column="w",
        trace_atol=2e-5,
    ),
    OptimizationLifecycleCase(
        case_id="negbin_est_ml_newton_joint_theta_cr",
        formula='y ~ s(x0, bs="cr", k=8)',
        family={"name": "negbin", "theta": 1.8, "estimate_theta": True},
        mgcv_family="negbin_est:1.8",
        method="ML",
        optimizer="newton",
        smoothing_optimizer="outer_newton",
        data_factory=_make_negbin_data,
        skip_coef_comparison=True,
        trace_atol=2e-5,
    ),
    OptimizationLifecycleCase(
        case_id="negbin_est_ml_bfgs_joint_theta_cr",
        formula='y ~ s(x0, bs="cr", k=8)',
        family={"name": "negbin", "theta": 1.8, "estimate_theta": True},
        mgcv_family="negbin_est:1.8",
        method="ML",
        optimizer="bfgs",
        smoothing_optimizer="bfgs",
        data_factory=_make_negbin_data,
        skip_coef_comparison=True,
        trace_atol=3e-5,
        trace_sp_atol=3e-5,
        cov_rtol=1e-4,
        cov_atol=1e-6,
        scalar_atol=1e-3,
    ),
    OptimizationLifecycleCase(
        case_id="negbin_est_reml_optim_joint_theta_cr",
        formula='y ~ s(x0, bs="cr", k=8)',
        family={"name": "negbin", "theta": 1.8, "estimate_theta": True},
        mgcv_family="negbin_est:1.8",
        method="REML",
        optimizer="optim",
        smoothing_optimizer="optim",
        data_factory=_make_negbin_data,
        skip_coef_comparison=True,
        # L-BFGS-B path drift vs R's optim accumulates over the joint
        # (log sp, log theta) trace; late-row log-sp/criterion values agree
        # only to ~4e-5 absolute (relative ~2e-6) across BLAS builds.
        trace_atol=1e-4,
        trace_sp_atol=1e-4,
        cov_rtol=1e-4,
        cov_atol=1e-6,
        scalar_atol=1e-3,
    ),
    OptimizationLifecycleCase(
        case_id="negbin_est_reml_bfgs_joint_theta_cr",
        formula='y ~ s(x0, bs="cr", k=8)',
        family={"name": "negbin", "theta": 1.8, "estimate_theta": True},
        mgcv_family="negbin_est:1.8",
        method="REML",
        optimizer="bfgs",
        smoothing_optimizer="bfgs",
        data_factory=_make_negbin_data,
        skip_coef_comparison=True,
        trace_atol=3e-5,
        trace_sp_atol=3e-5,
        cov_rtol=1e-4,
        cov_atol=1e-6,
        scalar_atol=1e-3,
    ),
]


__all__ = ["OPTIMIZATION_LIFECYCLE_CASES", "OptimizationLifecycleCase"]


OPTIMIZATION_LIFECYCLE_CASES.extend(
    [
        OptimizationLifecycleCase(
            case_id="poisson_reml_efs_tensor_cr",
            formula='y ~ te(x0, x1, bs=["cr", "cr"], k=[6, 6])',
            family="poisson",
            mgcv_family="poisson",
            method="REML",
            optimizer="efs",
            smoothing_optimizer="efs",
            data_factory=_make_poisson_data,
            skip_coef_comparison=True,
            trace_atol=3e-5,
            trace_sp_atol=3e-5,
            cov_rtol=1e-4,
            cov_atol=1e-6,
            scalar_atol=1e-3,
        ),
        OptimizationLifecycleCase(
            case_id="binomial_reml_bfgs_cr",
            formula='y ~ s(x0, bs="cr", k=8)',
            family="binomial",
            mgcv_family="binomial",
            method="REML",
            optimizer="bfgs",
            smoothing_optimizer="bfgs",
            data_factory=_make_binomial_data,
            skip_coef_comparison=True,
            trace_atol=3e-5,
            trace_sp_atol=3e-5,
            cov_rtol=1e-4,
            cov_atol=1e-6,
            scalar_atol=1e-3,
        ),
        OptimizationLifecycleCase(
            case_id="gamma_reml_bfgs_joint_scale_cr",
            formula='y ~ s(x0, bs="cr", k=8)',
            family="gamma",
            mgcv_family="gamma",
            method="REML",
            optimizer="bfgs",
            smoothing_optimizer="bfgs",
            data_factory=_make_gamma_data,
            trace_atol=5e-7,
            trace_sp_atol=5e-7,
        ),
        OptimizationLifecycleCase(
            case_id="gamma_reml_optim_joint_scale_cr",
            formula='y ~ s(x0, bs="cr", k=8)',
            family="gamma",
            mgcv_family="gamma",
            method="REML",
            optimizer="optim",
            smoothing_optimizer="optim",
            data_factory=_make_gamma_data,
            trace_atol=3e-5,
            trace_sp_atol=3e-5,
            cov_rtol=1e-4,
            cov_atol=1e-6,
            scalar_atol=1e-3,
        ),
        OptimizationLifecycleCase(
            case_id="negbin_fixed_theta_reml_newton_cr",
            formula='y ~ s(x0, bs="cr", k=8)',
            family={"name": "negbin", "theta": 1.8},
            mgcv_family="negbin:1.8",
            method="REML",
            optimizer="newton",
            smoothing_optimizer="outer_newton",
            data_factory=_make_negbin_data,
            skip_coef_comparison=True,
            trace_atol=3e-5,
            trace_sp_atol=3e-5,
            cov_rtol=1e-4,
            cov_atol=1e-6,
            scalar_atol=1e-3,
        ),
        OptimizationLifecycleCase(
            case_id="gaussian_reml_newton_random_effect",
            formula='y ~ s(f, bs="re")',
            family="gaussian",
            mgcv_family="gaussian",
            method="REML",
            optimizer="newton",
            smoothing_optimizer="outer_newton",
            data_factory=_coverage_make_random_effect_data_noisy,
            trace_atol=3e-5,
            trace_sp_atol=3e-5,
            cov_rtol=1e-4,
            cov_atol=1e-6,
            scalar_atol=1e-3,
        ),
    ]
)


# Additional supported lifecycle cases. This is intentionally not exhaustive.
OPTIMIZATION_LIFECYCLE_CASES.extend(
    [
        OptimizationLifecycleCase(
            case_id="gaussian_ml_newton_two_cr",
            formula='y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
            family="gaussian",
            mgcv_family="gaussian",
            method="ML",
            optimizer="newton",
            smoothing_optimizer="outer_newton",
            data_factory=_make_gaussian_data,
            trace_atol=3e-5,
        ),
        OptimizationLifecycleCase(
            case_id="binomial_ml_efs_cr",
            formula='y ~ s(x0, bs="cr", k=8)',
            family="binomial",
            mgcv_family="binomial",
            method="ML",
            optimizer="efs",
            smoothing_optimizer="efs",
            data_factory=_make_binomial_data,
            skip_coef_comparison=True,
            trace_atol=3e-5,
        ),
        OptimizationLifecycleCase(
            case_id="gaussian_reml_newton_fs_xt_ps",
            formula='y ~ s(f, x, bs="fs", xt=list(bs="ps", m=2, k=7))',
            family="gaussian",
            mgcv_family="gaussian",
            method="REML",
            optimizer="newton",
            smoothing_optimizer="outer_newton",
            data_factory=_coverage_make_random_effect_data_noisy,
            skip_coef_comparison=True,
            trace_atol=5e-5,
            # The two fs null-space penalties (sp indices 1, 2) are assigned to
            # nat.param null columns whose order is roundoff-indeterminate in
            # mgcv itself (see registry field docs). The base ps ignores k=/m=
            # inside xt exactly as upstream, so p0=10, range rank 8: null sp j
            # owns public coefficient columns 1 + level*10 + 8 + j.
            exchangeable_sp_groups=((1, 2),),
            exchangeable_sp_coef_cols=(((9, 19, 29), (10, 20, 30)),),
            # Vc's smoothing-uncertainty correction runs through the inverse
            # outer Hessian of the flat null directions, so it inherits the
            # branch of the null-order indeterminacy: mgcv on row-permuted
            # data (identical model) moves its own Vc intercept diagonal by
            # rel 6.3e-4, and NAMpy's value equals mgcv's other branch to 7
            # digits (0.06957781 vs 0.0695778077; Vp branch-invariant at
            # 2e-12). Verified 2026-08-15 in the fs null-order probes.
            compare_unconditional=False,
        ),
        OptimizationLifecycleCase(
            case_id="gaussian_reml_newton_re_custom_xt",
            formula='y ~ s(f, bs="re", xt=list(S=list(diag(3)), rank=c(3)))',
            family="gaussian",
            mgcv_family="gaussian",
            method="REML",
            optimizer="newton",
            smoothing_optimizer="outer_newton",
            data_factory=_coverage_make_random_effect_data_noisy,
            skip_coef_comparison=True,
            trace_atol=5e-5,
        ),
    ]
)


def _coverage_make_fs_lifecycle_data(seed=1711, n=180):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    levels = np.array(["a", "b", "c"], dtype=object)
    f = rng.choice(levels, size=n)
    x = rng.uniform(-1.5, 1.5, size=n)
    y = np.sin(x) + np.array([{"a": -0.25, "b": 0.2, "c": 0.45}[str(v)] for v in f])
    y = y + rng.normal(scale=0.08, size=n)
    return pd.DataFrame({"y": y, "f": f, "x": x})


for _idx, _case in enumerate(OPTIMIZATION_LIFECYCLE_CASES):
    if _case.case_id == "gaussian_reml_newton_fs_xt_ps":
        OPTIMIZATION_LIFECYCLE_CASES[_idx] = _optimization_lifecycle_replace(
            _case,
            data_factory=_coverage_make_fs_lifecycle_data,
        )
        break
