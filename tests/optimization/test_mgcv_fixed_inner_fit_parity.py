from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam.fit.backends import (
    solve_gaussian_given_smoothing,
    solve_pirls_given_smoothing,
)
from nampy.gam.fit.selection.criteria.dispatch import (
    criterion_gradient,
    criterion_hessian,
    criterion_value,
)
from nampy.gam.fit.selection.criteria.gaussian_dyn import (
    criterion_ml_reml_gaussian_dynamic_profiled,
)
from nampy.gam.fit.selection.criteria.pirls.derivatives import _gdi1_kernel
from nampy.gam.fit.solvers.general_family import newton as general_newton
from nampy.gam.fit.solvers.general_family.fixed_smoothing import (
    build_general_family_setup_state,
    run_general_family_fixed_smoothing,
    sl_initial_repara,
)
from nampy.gam.linalg.reindexing import permute_rows, restore_dropped_rows
from tests._paths import PARITY_DIR, REPO_ROOT
from tests.families.test_general_family_mgcv_parity import (
    GAULSS_FORMULA,
    _gaulss_data,
)
from tests.mgcv_parity_utils import (
    _fit_nampy_model_fixed_sp,
    _make_gaussian_data,
    _make_poisson_data,
)
from tests.reference_fixtures import load_reference, reference_key, save_reference

R_SCRIPT = shutil.which("Rscript")
MGCV_FIXED_SP_MAGIC_SCRIPT = PARITY_DIR / "mgcv_fixed_sp_magic.R"
MGCV_FIXED_SP_FIT3_SCRIPT = PARITY_DIR / "mgcv_fixed_sp_fit3.R"
MGCV_FIXED_SP_FIT5_SCRIPT = PARITY_DIR / "mgcv_fixed_sp_fit5.R"

pytestmark = [
    pytest.mark.method_fixed,
    pytest.mark.surface_derivatives,
    pytest.mark.surface_regression,
]


def _run_r_parity_script(script_path: Path, data, *args):
    key = reference_key(
        "fixed_inner_fit",
        {
            "script": script_path.name,
            "data": data.to_csv(index=False),
            "args": list(args),
        },
    )
    cached = load_reference("mgcv", key)
    if cached is not None:
        return cached
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
        result = json.loads(json_path.read_text(encoding="utf-8"))
        save_reference("mgcv", key, result)
        return result


def _run_mgcv_magic_fixed_sp(data, formula: str, sp: np.ndarray):
    return _run_r_parity_script(
        MGCV_FIXED_SP_MAGIC_SCRIPT,
        data,
        formula,
        json.dumps(np.asarray(sp, dtype=np.float64).tolist()),
    )


def _run_reference_fit3_fixed_sp(
    data,
    formula: str,
    family: str,
    sp: np.ndarray,
    *,
    score_type: str = "REML",
):
    return _run_r_parity_script(
        MGCV_FIXED_SP_FIT3_SCRIPT,
        data,
        formula,
        family,
        json.dumps(np.asarray(sp, dtype=np.float64).tolist()),
        str(score_type).upper(),
    )


def _run_reference_fit5_fixed_sp(
    data,
    formula,
    family: str,
    sp: np.ndarray,
    *,
    score_type: str = "REML",
):
    return _run_r_parity_script(
        MGCV_FIXED_SP_FIT5_SCRIPT,
        data,
        str(formula),
        family,
        json.dumps(np.asarray(sp, dtype=np.float64).tolist()),
        str(score_type).upper(),
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


def _make_linked_id_univariate_data(seed=1501, n=180):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.8, 1.8, size=n)
    y = np.sin(1.1 * x0) + 0.4 * np.cos(0.8 * x1) + rng.normal(scale=0.12, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_linked_id_cyclic_data(seed=1502, n=180):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x1 = rng.uniform(0.0, 2.0 * np.pi, size=n)
    y = np.sin(x0) + 0.35 * np.cos(1.5 * x1) + rng.normal(scale=0.08, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_linked_id_numeric_by_data(seed=1503, n=200):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.7, 1.7, size=n)
    z = rng.uniform(0.5, 1.5, size=n)
    y = z * (np.sin(x0) - 0.25 * np.cos(x1)) + rng.normal(scale=0.08, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1, "z": z})


def _make_gaulss_fs_data():
    rng = np.random.default_rng(248)
    n = 132
    row = np.arange(n)
    x0 = rng.uniform(-1.5, 1.5, size=n)
    f = np.asarray(["a", "b", "c"])[row % 3]
    f1 = np.asarray(["u", "v", "w"])[row % 3]
    f2 = np.asarray(["left", "right"])[(row // 3) % 2]
    f_effect = np.asarray(
        [{"a": -0.3, "b": 0.05, "c": 0.25}[value] for value in f]
    )
    cell_effect = np.asarray(
        [
            {
                ("u", "left"): -0.16,
                ("u", "right"): 0.08,
                ("v", "left"): 0.18,
                ("v", "right"): -0.1,
                ("w", "left"): 0.06,
                ("w", "right"): -0.04,
            }[(left, right)]
            for left, right in zip(f1, f2, strict=True)
        ]
    )
    signal = 0.3 * np.sin(1.4 * x0) + f_effect + cell_effect
    sigma = np.exp(-0.45 + 0.12 * x0)
    y = rng.normal(signal, sigma, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "f": f, "f1": f1, "f2": f2})


LINKED_ID_FIXED_SP_CASES = [
    pytest.param(
        "linked_cr",
        _make_linked_id_univariate_data,
        'y ~ s(x0, bs="cr", k=6, id="g") + s(x1, bs="cr", k=6, id="g")',
        np.array([0.7], dtype=np.float64),
        False,
        1e-8,
        True,
        id="linked_cr",
    ),
    pytest.param(
        "linked_cs",
        _make_linked_id_univariate_data,
        'y ~ s(x0, bs="cs", k=6, id="g") + s(x1, bs="cs", k=6, id="g")',
        np.array([0.7], dtype=np.float64),
        False,
        3e-4,
        True,
        id="linked_cs",
    ),
    pytest.param(
        "linked_ps_m_ordered",
        _make_linked_id_univariate_data,
        'y ~ s(x0, bs="ps", k=8, m=[2, 3], id="g")'
        ' + s(x1, bs="ps", k=8, m=[2, 3], id="g")',
        np.array([0.55], dtype=np.float64),
        False,
        5e-8,
        True,
        id="linked_ps_m_ordered",
    ),
    pytest.param(
        "linked_tp",
        _make_linked_id_univariate_data,
        'y ~ s(x0, bs="tp", k=8, id="g") + s(x1, bs="tp", k=8, id="g")',
        np.array([0.4], dtype=np.float64),
        False,
        5e-7,
        # Thin-plate eigenvector signs are not identified; eta, mu, and
        # deviance below are the strict fixed-sp behavioral contract.
        False,
        id="linked_tp",
    ),
    pytest.param(
        "linked_ts",
        _make_linked_id_univariate_data,
        'y ~ s(x0, bs="ts", k=8, id="g") + s(x1, bs="ts", k=8, id="g")',
        np.array([0.4], dtype=np.float64),
        False,
        2e-4,
        False,
        id="linked_ts",
    ),
    pytest.param(
        "linked_cc",
        _make_linked_id_cyclic_data,
        'y ~ s(x0, bs="cc", k=6, id="g") + s(x1, bs="cc", k=6, id="g")',
        np.array([0.65], dtype=np.float64),
        False,
        1e-8,
        True,
        id="linked_cc",
    ),
    pytest.param(
        "linked_cr_numeric_by",
        _make_linked_id_numeric_by_data,
        'y ~ s(x0, by=z, bs="cr", k=6, id="g")' ' + s(x1, by=z, bs="cr", k=6, id="g")',
        np.array([0.75], dtype=np.float64),
        False,
        1e-10,
        False,
        id="linked_cr_numeric_by",
    ),
    pytest.param(
        "linked_cr_incompatible_k",
        _make_linked_id_univariate_data,
        'y ~ s(x0, bs="cr", k=6, id="g") + s(x1, bs="cr", k=8, id="g")',
        np.array([0.7], dtype=np.float64),
        False,
        5e-8,
        True,
        id="linked_cr_incompatible_k",
    ),
]


def test_gaussian_magic_fixed_sp_state_matches_mgcv():
    """Verify that gaussian magic fixed sp state matches mgcv."""
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
    """Verify that poisson gam fit3 fixed sp inner state matches mgcv."""
    data = _make_poisson_data(seed=789, n=180)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
    sp = np.array([0.6, 1.3], dtype=np.float64)

    gam = _fit_nampy_model_fixed_sp(data, formula, "poisson", sp)
    y = gam.family.validate_y(gam.y_)
    sol = solve_pirls_given_smoothing(gam, y, sp)
    kernel = _gdi1_kernel(gam, y, sol, sp, method="REML")
    expected = _run_reference_fit3_fixed_sp(data, formula, "poisson", sp)

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

    crit_atol = 5e-6 if "ti(" in formula else 5e-8
    deriv_atol = 5e-6 if "ti(" in formula else 5e-8
    assert actual_reml == pytest.approx(float(expected["REML"]), abs=crit_atol)
    _assert_allclose(actual_grad, expected["REML1"], atol=deriv_atol)
    _assert_allclose(actual_hess, expected["REML2"], atol=deriv_atol)


@pytest.mark.parametrize(
    ("score_type", "method", "score_key", "grad_key", "hess_key"),
    [
        ("GCV", "gcv", "GCV", "GCV1", "GCV2"),
        ("UBRE", "ubre", "UBRE", "UBRE1", "UBRE2"),
    ],
    ids=["gcv", "ubre"],
)
def test_poisson_gam_fit3_gcv_ubre_derivatives_match_mgcv(
    score_type,
    method,
    score_key,
    grad_key,
    hess_key,
):
    """Verify that poisson gam.fit3 GCV/UBRE derivatives match mgcv."""
    data = _make_poisson_data(seed=789, n=180)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
    sp = np.array([0.6, 1.3], dtype=np.float64)

    gam = _fit_nampy_model_fixed_sp(data, formula, "poisson", sp)
    y = gam.family.validate_y(gam.y_)
    expected = _run_reference_fit3_fixed_sp(
        data,
        formula,
        "poisson",
        sp,
        score_type=score_type,
    )

    actual_score = float(criterion_value(gam, y, np.log(sp), method=method))
    actual_grad = np.asarray(criterion_gradient(gam, y, np.log(sp), method=method))
    actual_hess = np.asarray(criterion_hessian(gam, y, np.log(sp), method=method))

    assert actual_score == pytest.approx(float(expected[score_key]), abs=5e-8)
    _assert_allclose(actual_grad, expected[grad_key], atol=5e-8)
    _assert_allclose(actual_hess, expected[hess_key], atol=5e-8)


def test_gaulss_gam_fit5_fixed_sp_inner_state_matches_mgcv():
    """Verify that gaulss gam fit5 fixed sp inner state matches mgcv."""
    data = _gaulss_data(seed=11, n=140)
    sp = np.array([0.9], dtype=np.float64)

    gam = _fit_nampy_model_fixed_sp(data, GAULSS_FORMULA, "gaulss", sp)
    y = gam.family.validate_y(gam.y_)
    run = run_general_family_fixed_smoothing(
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
    expected = _run_reference_fit5_fixed_sp(data, GAULSS_FORMULA, "gaulss", sp)

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


def test_gaulss_fs_gam_fit5_fixed_sp_inner_state_matches_mgcv():
    """Keep the fs penalty block aligned through initialization and gam.fit5."""
    data = _make_gaulss_fs_data()
    formula = ['y ~ s(f, x0, bs="fs", k=5, xt="cr")', "~ 1"]
    # The two null-space vectors are an indeterminate eigenspace. Using the
    # same fixed value for both makes the physical penalty invariant to their
    # legal orientation/permutation difference across LAPACK implementations.
    sp = np.array([0.8, 0.8, 0.8], dtype=np.float64)

    gam = _fit_nampy_model_fixed_sp(data, formula, "gaulss", sp)
    y = gam.family.validate_y(gam.y_)
    setup = build_general_family_setup_state(gam, sp, score_type="REML")
    rp = general_newton._sl_ldetS(
        setup.Sl,
        rho=setup.log_sp,
        fixed=np.zeros(sp.size, dtype=bool),
        np_=setup.X_initial.shape[1],
        root=True,
        Stot=True,
        deriv=2,
    )
    X_fit = general_newton._sl_repara(rp["rp"], setup.X_initial)
    E_fit = general_newton._PenaltyRoot(rp["E"], use_unscaled=True)
    start = gam.family.initialize(
        y,
        X_fit,
        setup.jj,
        offset=setup.offset_list,
        weights=gam.prior_weights_,
        E=E_fit,
    )
    actual_start_eta = np.column_stack(
        [X_fit[:, jj] @ start[jj] for jj in setup.jj]
    )

    run = run_general_family_fixed_smoothing(
        gam,
        y,
        sp,
        weights=gam.prior_weights_,
        deriv=2,
        score_type="REML",
    )
    actual_eta = _fit5_linear_predictors(
        run["setup"], run["fit"], run["offset_list"]
    )
    expected = _run_reference_fit5_fixed_sp(data, formula, "gaulss", sp)

    _assert_allclose(
        actual_start_eta,
        np.column_stack(expected["start_linear_predictors"]),
        atol=1e-8,
    )
    _assert_allclose(actual_eta, expected["linear_predictors"], atol=1e-8)


@pytest.mark.method_reml
@pytest.mark.family_gaussian
@pytest.mark.parametrize(
    ("_case_id", "data_factory", "formula", "sp", "select", "atol", "compare_coef"),
    LINKED_ID_FIXED_SP_CASES,
)
def test_linked_id_gaussian_magic_fixed_sp_matches_mgcv_supported_bases(
    _case_id, data_factory, formula, sp, select, atol, compare_coef
):
    """Verify that linked id gaussian magic fixed sp matches mgcv supported bases."""
    data = data_factory()
    gam = _fit_nampy_model_fixed_sp(
        data,
        formula,
        "gaussian",
        sp,
        select=select,
    )
    y = gam.family.validate_y(gam.y_)
    sol = solve_gaussian_given_smoothing(gam, y, sp)
    expected = _run_mgcv_magic_fixed_sp(data, formula, sp)

    if compare_coef:
        _assert_allclose(sol.coef_full, expected["coefficients"], atol=atol)
    _assert_allclose(sol.eta, expected["linear_predictors"], atol=atol)
    _assert_allclose(sol.mu, expected["fitted_values"], atol=atol)
    _assert_allclose(sol.deviance, expected["deviance"], atol=atol)
