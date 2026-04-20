from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nampy.gam.fit.linalg.matrix_reindexing import permute_rows, restore_dropped_rows
from nampy.gam.fit.solve_ops import (
    solve_gaussian_given_smoothing,
    solve_pirls_given_smoothing,
)
from nampy.gam.fit.solvers.general_fit5 import _run_general_fit5, sl_initial_repara
from nampy.gam.smoothing_selection.criteria.dispatch import (
    criterion_gradient,
    criterion_hessian,
    criterion_value,
)
from nampy.gam.smoothing_selection.criteria.gaussian_dyn import (
    criterion_ml_reml_gaussian_dynamic_profiled,
)
from nampy.gam.smoothing_selection.criteria.pirls_deriv import _gdi1_kernel
from tests._paths import PARITY_DIR, REPO_ROOT
from tests.families.test_general_family_mgcv_parity import GAULSS_FORMULA, _gaulss_data
from tests.mgcv_parity_utils import (
    _fit_nampy_model_fixed_sp,
    _make_gaussian_data,
    _make_poisson_data,
)

R_SCRIPT = shutil.which("Rscript")
MGCV_FIXED_SP_MAGIC_SCRIPT = PARITY_DIR / "mgcv_fixed_sp_magic.R"
MGCV_FIXED_SP_FIT3_SCRIPT = PARITY_DIR / "mgcv_fixed_sp_fit3.R"
MGCV_FIXED_SP_FIT5_SCRIPT = PARITY_DIR / "mgcv_fixed_sp_fit5.R"

pytestmark = [
    pytest.mark.method_fixed,
    pytest.mark.surface_derivatives,
    pytest.mark.surface_regression,
    pytest.mark.skipif(R_SCRIPT is None, reason="Rscript required for mgcv parity"),
]


def _run_r_parity_script(script_path: Path, data, *args):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "out.json"
        data.to_csv(csv_path, index=False)
        subprocess.run(
            [R_SCRIPT, str(script_path), str(csv_path), str(json_path), *args],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def _run_mgcv_magic_fixed_sp(data, formula: str, sp: np.ndarray):
    return _run_r_parity_script(
        MGCV_FIXED_SP_MAGIC_SCRIPT,
        data,
        formula,
        json.dumps(np.asarray(sp, dtype=np.float64).tolist()),
    )


def _run_mgcv_fit3_fixed_sp(data, formula: str, family: str, sp: np.ndarray):
    return _run_r_parity_script(
        MGCV_FIXED_SP_FIT3_SCRIPT,
        data,
        formula,
        family,
        json.dumps(np.asarray(sp, dtype=np.float64).tolist()),
    )


def _run_mgcv_fit5_fixed_sp(data, formula, family: str, sp: np.ndarray):
    return _run_r_parity_script(
        MGCV_FIXED_SP_FIT5_SCRIPT,
        data,
        str(formula),
        family,
        json.dumps(np.asarray(sp, dtype=np.float64).tolist()),
    )


def _expand_dbeta_to_original_space(kernel) -> np.ndarray:
    if len(kernel.ift.dbeta) == 0:
        q = int(np.asarray(kernel.current.canonical.T, dtype=np.float64).shape[0])
        return np.empty((q, 0), dtype=np.float64)

    packed = np.column_stack(
        [np.asarray(v, dtype=np.float64) for v in kernel.ift.dbeta]
    )
    unpivot = permute_rows(packed, kernel.current.pivot1, reverse=True)
    full = restore_dropped_rows(
        unpivot,
        int(np.asarray(kernel.current.canonical.T, dtype=np.float64).shape[0]),
        kernel.current.dropped_column_indices,
    )
    return np.asarray(kernel.current.canonical.T, dtype=np.float64) @ full


def _assert_allclose(actual, expected, *, atol=1e-8, err_msg=""):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=0.0,
        atol=atol,
        err_msg=err_msg,
    )


def _fit5_linear_predictors(setup, fit, offset_list):
    coef = np.asarray(fit["coef"], dtype=np.float64)
    eta_cols = []
    for k, jj in enumerate(setup.jj):
        jj = np.asarray(jj, dtype=np.int64)
        eta_k = np.asarray(setup.X_initial[:, jj] @ coef[jj], dtype=np.float64)
        if offset_list is not None and k < len(offset_list):
            off_k = offset_list[k]
            if off_k is not None:
                eta_k = eta_k + np.asarray(off_k, dtype=np.float64)
        eta_cols.append(eta_k)
    return np.column_stack(eta_cols)


def test_gaussian_magic_fixed_sp_state_matches_mgcv():
    data = _make_gaussian_data(seed=1401, n=180)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
    sp = np.array([0.45, 2.15], dtype=np.float64)

    gam = _fit_nampy_model_fixed_sp(data, formula, "gaussian", sp)
    y = gam.family.validate_y(gam.y_)
    sol = solve_gaussian_given_smoothing(gam, y, sp)
    expected = _run_mgcv_magic_fixed_sp(data, formula, sp)

    _assert_allclose(sol.coef_full, expected["coefficients"], atol=1e-10)
    _assert_allclose(sol.eta, expected["linear_predictors"], atol=1e-10)
    _assert_allclose(sol.mu, expected["fitted_values"], atol=1e-10)
    _assert_allclose(sol.deviance, expected["deviance"], atol=1e-10)
    _assert_allclose(sol.working_weights, expected["working_weights"], atol=1e-12)
    _assert_allclose(sol.fisher_weights, expected["weights"], atol=1e-12)
    _assert_allclose(sol.working_response, expected["working_response"], atol=1e-12)

    actual_reml = float(
        criterion_ml_reml_gaussian_dynamic_profiled(
            gam,
            y,
            np.log(sp),
            method="REML",
        )
    )
    assert actual_reml == pytest.approx(float(expected["reml"]), abs=1e-8)


def test_poisson_gam_fit3_fixed_sp_inner_state_matches_mgcv():
    data = _make_poisson_data(seed=789, n=180)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
    sp = np.array([0.6, 1.3], dtype=np.float64)

    gam = _fit_nampy_model_fixed_sp(data, formula, "poisson", sp)
    y = gam.family.validate_y(gam.y_)
    sol = solve_pirls_given_smoothing(gam, y, sp)
    kernel = _gdi1_kernel(gam, y, sol, sp, method="REML")
    expected = _run_mgcv_fit3_fixed_sp(data, formula, "poisson", sp)

    _assert_allclose(sol.coef_full, expected["coefficients"], atol=1e-8)
    _assert_allclose(sol.eta, expected["linear_predictors"], atol=1e-8)
    _assert_allclose(sol.mu, expected["fitted_values"], atol=1e-8)
    _assert_allclose(sol.deviance, expected["deviance"], atol=1e-8)
    _assert_allclose(sol.fisher_weights, expected["weights"], atol=5e-8)
    _assert_allclose(sol.working_weights, expected["working_weights"], atol=5e-8)
    _assert_allclose(sol.working_response, expected["working_response"], atol=2e-7)

    _assert_allclose(kernel.dVkk, expected["dVkk"], atol=5e-8)
    _assert_allclose(
        _expand_dbeta_to_original_space(kernel), expected["db_drho"], atol=5e-8
    )

    actual_reml = float(criterion_value(gam, y, np.log(sp), method="reml"))
    actual_grad = np.asarray(criterion_gradient(gam, y, np.log(sp), method="reml"))
    actual_hess = np.asarray(criterion_hessian(gam, y, np.log(sp), method="reml"))

    assert actual_reml == pytest.approx(float(expected["REML"]), abs=5e-8)
    _assert_allclose(actual_grad, expected["REML1"], atol=5e-8)
    _assert_allclose(actual_hess, expected["REML2"], atol=5e-8)


def test_gaulss_gam_fit5_fixed_sp_inner_state_matches_mgcv():
    data = _gaulss_data(seed=11, n=140)
    sp = np.array([0.9], dtype=np.float64)

    gam = _fit_nampy_model_fixed_sp(data, GAULSS_FORMULA, "gaulss", sp)
    y = gam.family.validate_y(gam.y_)
    run = _run_general_fit5(
        gam,
        y,
        sp,
        weights=gam.prior_weights_,
        deriv=2,
        score_type="REML",
    )
    fit = run["fit"]
    setup = run["setup"]
    actual_coef_full = np.asarray(
        sl_initial_repara(
            setup.Sl,
            np.asarray(fit["coef"], dtype=np.float64),
            inverse=True,
            both_sides=False,
            cov=False,
        ),
        dtype=np.float64,
    )
    actual_eta = _fit5_linear_predictors(setup, fit, run["offset_list"])
    actual_fitted = np.asarray(gam.family.predict(eta=actual_eta), dtype=np.float64)
    actual_db_drho_full = np.column_stack(
        [
            np.asarray(
                sl_initial_repara(
                    setup.Sl,
                    np.asarray(fit["db_drho"], dtype=np.float64)[:, i],
                    inverse=True,
                    both_sides=False,
                    cov=False,
                ),
                dtype=np.float64,
            )
            for i in range(np.asarray(fit["db_drho"], dtype=np.float64).shape[1])
        ]
    )
    expected = _run_mgcv_fit5_fixed_sp(data, GAULSS_FORMULA, "gaulss", sp)

    _assert_allclose(actual_coef_full, expected["coefficients_full"], atol=1e-8)
    _assert_allclose(actual_eta, expected["linear_predictors"], atol=1e-8)
    _assert_allclose(actual_fitted, expected["fitted_values"], atol=1e-8)
    _assert_allclose(-2.0 * float(fit["l"]), expected["deviance"], atol=1e-8)
    _assert_allclose(actual_db_drho_full, expected["db_drho_full"], atol=1e-8)
    assert float(fit["REML"]) == pytest.approx(float(expected["REML"]), abs=1e-8)
    _assert_allclose(fit["REML1"], expected["REML1"], atol=1e-8)
    _assert_allclose(fit["REML2"], expected["REML2"], atol=1e-8)

    actual_reml = float(criterion_value(gam, y, np.log(sp), method="reml"))
    actual_grad = np.asarray(criterion_gradient(gam, y, np.log(sp), method="reml"))
    actual_hess = np.asarray(criterion_hessian(gam, y, np.log(sp), method="reml"))

    assert actual_reml == pytest.approx(float(expected["REML"]), abs=1e-8)
    _assert_allclose(actual_grad, expected["REML1"], atol=1e-8)
    _assert_allclose(actual_hess, expected["REML2"], atol=1e-8)
