from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from tests.families.test_general_family_mgcv_parity import _gaulss_two_smooth_data
from tests.mgcv_parity_utils import (
    _make_gamma_data,
    _make_negbin_data,
    _make_poisson_data,
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
        trace_atol=5e-7,
        gam_kwargs={"sp_log_bounds": (-80.0, 25.0)},
    ),
    OptimizationLifecycleCase(
        case_id="gaulss_reml_efs_two_cr",
        formula=['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"],
        family="gaulss",
        mgcv_family="gaulss",
        method="REML",
        optimizer="efs",
        smoothing_optimizer="efs",
        data_factory=_gaulss_two_smooth_data,
        compare_hat=False,
        exact_outer_info_trace=False,
        trace_atol=5e-6,
        cov_rtol=5e-5,
        cov_atol=5e-8,
        scalar_atol=5e-4,
    ),
    OptimizationLifecycleCase(
        case_id="gaulss_ml_newton_two_cr",
        formula=['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"],
        family="gaulss",
        mgcv_family="gaulss",
        method="ML",
        optimizer="newton",
        smoothing_optimizer="outer_newton",
        data_factory=_gaulss_two_smooth_data,
        compare_hat=False,
        exact_outer_info_trace=False,
        trace_atol=1e-6,
        cov_rtol=5e-5,
        cov_atol=5e-8,
        scalar_atol=5e-4,
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
        status="known_gap",
        known_gap_reason=(
            "Gamma joint REML trace still exposes PIRLS-inner rows instead of "
            "mgcv's joint log-scale outer rows."
        ),
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
        status="known_gap",
        known_gap_reason=(
            "Joint negbin outer trace still swaps the smoothing-parameter and "
            "log-theta row labels on the serialized branch trace."
        ),
    ),
]


__all__ = ["OPTIMIZATION_LIFECYCLE_CASES", "OptimizationLifecycleCase"]
