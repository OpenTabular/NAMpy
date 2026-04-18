import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nampy.gam import GAM
from nampy.gam.smoothing_selection.reparam import (
    build_estimate_gam_setup_state,
    gam_reparam,
)
from tests._paths import PARITY_DIR, REPO_ROOT
from tests.mgcv_parity_utils import _family_specs, _fit_nampy_model_fixed_sp
from tests.optimization.test_mgcv_preoptimization_blocks_parity import PREOPT_CASES

R_SCRIPT = shutil.which("Rscript")
MGCV_PREOPT_REPARAM_SCRIPT = PARITY_DIR / "mgcv_preoptimization_reparam.R"

REPARAM_CASE_IDS = {
    "gaussian_two_cr",
    "gaussian_linked_id_two_cr",
    "gaussian_select_true_two_cr",
    "gaussian_ti_two_dim",
    "gaussian_t2_full_false",
    "gaussian_fs",
    "gaussian_random_effect",
    "gaussian_numeric_by_cr",
    "gaussian_factor_by_cr",
    "binomial_two_cr",
    "poisson_numeric_by_cr",
    "gamma_ps_uni",
    "negbin_est_fixed_sp",
}

REPARAM_CASES = [case for case in PREOPT_CASES if case[0] in REPARAM_CASE_IDS]


def _normalize_family_name(family):
    if isinstance(family, dict):
        return str(family.get("name", "")).lower()
    return str(family).lower()


def test_reparam_case_matrix_covers_requested_surface():
    ids = {case[0] for case in REPARAM_CASES}
    assert ids == REPARAM_CASE_IDS

    families = {_normalize_family_name(case[3]) for case in REPARAM_CASES}
    assert families >= {"gaussian", "binomial", "poisson", "gamma", "negbin"}

    assert any(case[5] for case in REPARAM_CASES), "Missing select=True coverage."
    assert any('id="' in str(case[2]) for case in REPARAM_CASES)
    assert any("by=z" in str(case[2]) for case in REPARAM_CASES)
    assert any("by=f" in str(case[2]) for case in REPARAM_CASES)
    assert any(
        any(token in str(case[2]) for token in ("ti(", "t2(", 'bs="fs"'))
        for case in REPARAM_CASES
    )
    assert any('bs="re"' in str(case[2]) for case in REPARAM_CASES)


def _run_mgcv_preoptimization_reparam(data, formula, family, method, *, select=False):
    family_nampy, family_token = _family_specs(family)
    del family_nampy

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "preopt_reparam.json"
        data.to_csv(csv_path, index=False)
        subprocess.run(
            [
                R_SCRIPT,
                str(MGCV_PREOPT_REPARAM_SCRIPT),
                str(csv_path),
                str(json_path),
                formula,
                family_token,
                method,
                "true" if select else "false",
            ],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def _fit_nampy_reparameterization_model(data, formula, family, fit_sp, *, select=False):
    fit_sp = np.asarray(fit_sp, dtype=np.float64).ravel()
    if fit_sp.size > 0:
        return _fit_nampy_model_fixed_sp(
            data,
            formula,
            family,
            fit_sp,
            select=select,
        )

    family_nampy, _ = _family_specs(family)
    gam = GAM(
        family=family_nampy,
        formula=formula,
        select=select,
        optimize_smoothing=False,
        smoothing_method="fixed",
    )
    gam.fit(data=data)
    return gam


def _as_matrix_list(value):
    return [np.asarray(item, dtype=np.float64) for item in (value or [])]


def _assert_projector_equal(actual, expected, *, atol=1e-10):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    np.testing.assert_allclose(
        actual @ actual.T,
        expected @ expected.T,
        rtol=0.0,
        atol=atol,
    )


def _assert_root_gram_equal(actual, expected, *, atol=1e-10):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    np.testing.assert_allclose(
        actual.T @ actual,
        expected.T @ expected,
        rtol=0.0,
        atol=atol,
    )


def _assert_root_singular_values_equal(actual, expected, *, atol=1e-10):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    actual_sv = np.sort(np.linalg.svd(actual, compute_uv=False))
    expected_sv = np.sort(np.linalg.svd(expected, compute_uv=False))
    np.testing.assert_allclose(actual_sv, expected_sv, rtol=0.0, atol=atol)


def _assert_u1_subspaces_equal(actual, expected, *, q_range, atol=1e-10):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape

    q_range = int(q_range)
    if q_range > 0:
        _assert_projector_equal(actual[:, :q_range], expected[:, :q_range], atol=atol)

    q_null = int(actual.shape[1]) - q_range
    if q_null > 0:
        _assert_projector_equal(actual[:, q_range:], expected[:, q_range:], atol=atol)


def _assert_symmetric_spectrum_equal(actual, expected, *, atol=1e-10):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    actual_eval = np.linalg.eigvalsh(0.5 * (actual + actual.T))
    expected_eval = np.linalg.eigvalsh(0.5 * (expected + expected.T))
    # `mgcv::gam.reparam()` uses the C `stableS` path upstream, while NAMpy
    # reconstructs the same surface in Python. Large penalty eigenvalues can
    # differ by a few ULPs, so compare with a tiny relative tolerance as well.
    np.testing.assert_allclose(actual_eval, expected_eval, rtol=2e-11, atol=atol)


def _current_log_sp_full(model, setup):
    sp_all = np.asarray(model.smoothing_params, dtype=np.float64).ravel()
    fixed_mask = (
        np.zeros(sp_all.shape, dtype=bool)
        if getattr(model, "smoothing_fixed_mask_", None) is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    free_sp = np.asarray(sp_all[~fixed_mask], dtype=np.float64)
    lsp0 = np.asarray(setup.lsp0, dtype=np.float64)
    if free_sp.size == 0:
        return lsp0.copy()
    if setup.L is None:
        return np.log(free_sp) + lsp0
    return np.asarray(
        np.asarray(setup.L, dtype=np.float64) @ np.log(free_sp) + lsp0,
        dtype=np.float64,
    )


def _assert_setup_reparam_surface(actual, expected, *, atol=1e-10):
    expected_E = np.asarray(expected["E"], dtype=np.float64)
    expected_Eb = np.asarray(expected["Eb"], dtype=np.float64)
    expected_U1 = np.asarray(expected["U1"], dtype=np.float64)
    expected_UrS = _as_matrix_list(expected.get("UrS", []))

    assert int(actual.Mp) == int(expected["Mp"])
    assert actual.E.shape == expected_E.shape
    assert actual.Eb.shape == expected_Eb.shape
    assert actual.U1.shape == expected_U1.shape
    assert len(actual.UrS) == len(expected_UrS)

    _assert_u1_subspaces_equal(
        actual.U1, expected_U1, q_range=expected_E.shape[0], atol=atol
    )
    _assert_root_gram_equal(actual.E, expected_E, atol=atol)
    _assert_root_gram_equal(actual.Eb, expected_Eb, atol=atol)
    for a_root, e_root in zip(actual.UrS, expected_UrS):
        _assert_root_gram_equal(a_root, e_root, atol=atol)


def _assert_gam_reparam_invariants(actual, expected, *, atol=1e-10):
    assert bool(actual["fixed_penalty"]) is bool(expected["fixed_penalty"])
    assert float(actual["det"]) == pytest.approx(float(expected["det"]), abs=atol)
    np.testing.assert_allclose(
        np.asarray(actual["det1"], dtype=np.float64),
        np.asarray(expected["det1"], dtype=np.float64),
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(actual["det2"], dtype=np.float64),
        np.asarray(expected["det2"], dtype=np.float64),
        rtol=0.0,
        atol=max(atol, 5e-10),
    )

    _assert_symmetric_spectrum_equal(actual["S"], expected["S"], atol=atol)
    _assert_root_singular_values_equal(actual["E"], expected["E"], atol=atol)

    expected_rS = _as_matrix_list(expected.get("rS", []))
    assert len(actual["rS"]) == len(expected_rS)
    for a_root, e_root in zip(actual["rS"], expected_rS):
        _assert_root_gram_equal(a_root, e_root, atol=atol)


@pytest.mark.parametrize(
    "case_id, data_factory, formula, family, method, select, _compare_design_space_only",
    REPARAM_CASES,
    ids=[case[0] for case in REPARAM_CASES],
)
def test_preoptimization_reparameterization_matches_mgcv(
    case_id,
    data_factory,
    formula,
    family,
    method,
    select,
    _compare_design_space_only,
):
    del case_id, _compare_design_space_only
    data = data_factory()
    expected = _run_mgcv_preoptimization_reparam(
        data,
        formula,
        family,
        method,
        select=select,
    )

    gam = _fit_nampy_reparameterization_model(
        data,
        formula,
        family,
        expected["fit_sp"],
        select=select,
    )
    actual_setup = build_estimate_gam_setup_state(gam)
    np.testing.assert_allclose(
        _current_log_sp_full(gam, actual_setup),
        np.asarray(expected["log_sp_full"], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    _assert_setup_reparam_surface(actual_setup, expected["setup"])

    actual_rp = gam_reparam(
        [np.asarray(root, dtype=np.float64) for root in actual_setup.UrS],
        _current_log_sp_full(gam, actual_setup),
        deriv=2,
    )
    _assert_gam_reparam_invariants(actual_rp, expected["gam_reparam"])
