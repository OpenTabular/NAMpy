"""Shared helpers for tests/parity/test_mgcv_snapshot_core_matrix.py cases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nampy.gam import GAM
from nampy.gam.parity import covariance_standard_errors


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    formula: str
    family: str | dict
    data_factory: callable
    select: bool = False
    weights_column: str | None = None
    # tp eigenvector signs are LAPACK-implementation-dependent; compare predictions instead.
    skip_coef_comparison: bool = False
    criterion_atol: float = 1e-4
    se_tol_scale: float = 1e-6


def _fit_nampy_snapshot(case: CaseSpec, data: pd.DataFrame):
    model = GAM(
        family=case.family,
        formula=case.formula,
        select=case.select,
        optimize_smoothing=True,
        smoothing_method="REML",
    )
    sample_weight = None
    if case.weights_column is not None:
        sample_weight = np.asarray(data[case.weights_column], dtype=np.float64)
    model.fit(data=data, sample_weight=sample_weight)
    return model.parity_snapshot(X=data, include_covariances=True)


def _assert_requested_parity(
    case: CaseSpec,
    actual_snapshot: dict,
    expected_snapshot: dict,
) -> None:
    if case.skip_coef_comparison:
        link_actual = np.asarray(
            actual_snapshot["predictions"]["link"], dtype=np.float64
        )
        link_expected = np.asarray(
            expected_snapshot["predictions"]["link"], dtype=np.float64
        )
        link_tol = 1e-4 * (1.0 + np.abs(link_actual))
        link_err = np.abs(link_actual - link_expected)
        assert np.all(link_err <= link_tol), (
            f"{case.case_id}: |link-link_mgcv| exceeded tolerance; "
            f"max_err={link_err.max():.3e}, max_tol={link_tol.max():.3e}"
        )
    else:
        beta = np.asarray(actual_snapshot["fit"]["coef_full"], dtype=np.float64)
        beta_mgcv = np.asarray(expected_snapshot["fit"]["coef_full"], dtype=np.float64)
        assert beta.shape == beta_mgcv.shape, f"{case.case_id}: beta shape mismatch"
        beta_tol = 1e-6 * (1.0 + np.abs(beta))
        beta_err = np.abs(beta - beta_mgcv)
        assert np.all(beta_err <= beta_tol), (
            f"{case.case_id}: |beta-beta_mgcv| exceeded tolerance; max_err={beta_err.max():.3e}, "
            f"max_tol={beta_tol.max():.3e}"
        )

    edf = float(actual_snapshot["fit"]["edf_total"])
    edf_mgcv = float(expected_snapshot["fit"]["edf_total"])
    assert (
        abs(edf - edf_mgcv) < 1e-4
    ), f"{case.case_id}: |edf-edf_mgcv|={abs(edf - edf_mgcv):.3e} >= 1e-4"

    reml = float(actual_snapshot["fit"]["criterion_value"])
    reml_mgcv = float(expected_snapshot["fit"]["criterion_value"])
    assert abs(reml - reml_mgcv) < float(case.criterion_atol), (
        f"{case.case_id}: |REML-REML_mgcv|={abs(reml - reml_mgcv):.3e} "
        f">= {float(case.criterion_atol):.3e}"
    )

    cov = np.asarray(actual_snapshot["fit"]["cov_bayes"], dtype=np.float64)
    cov_mgcv = np.asarray(expected_snapshot["fit"]["cov_bayes"], dtype=np.float64)
    assert cov.shape == cov_mgcv.shape, f"{case.case_id}: covariance shape mismatch"
    se = covariance_standard_errors(cov)
    se_mgcv = covariance_standard_errors(cov_mgcv)
    se_tol = float(case.se_tol_scale) * (1.0 + np.abs(se))
    se_err = np.abs(se - se_mgcv)
    assert np.all(se_err <= se_tol), (
        f"{case.case_id}: |se-se_mgcv| exceeded tolerance; max_err={se_err.max():.3e}, "
        f"max_tol={se_tol.max():.3e}"
    )
