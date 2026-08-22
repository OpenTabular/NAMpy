from __future__ import annotations

import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from nampy.gam import GAM
from nampy.gam.fit import covariance as covariance_module
from nampy.gam.fit.covariance import (
    build_bayes_and_freq_covariances,
    select_covariance_matrix,
)
from tests.mgcv_parity_utils import (
    _fit_nampy_model,
    _run_mgcv_snapshot,
    get_parity_case,
    make_parity_case_data,
)

R_SCRIPT = shutil.which("Rscript")

pytestmark = [
    pytest.mark.surface_output,
    pytest.mark.surface_regression,
]


def test_build_bayes_and_freq_covariances_symmetrizes_freq_branch_only():
    """
    Owner-contract coverage verifying that build bayes and freq covariances symmetrizes
    freq branch only.
    """
    scale = 2.0
    A_inv = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float64)
    XWX = np.array([[4.0, 1.0], [1.0, 5.0]], dtype=np.float64)

    Vp, Vf, H_coef = build_bayes_and_freq_covariances(scale, A_inv, XWX)

    A_inv_sym = 0.5 * (A_inv + A_inv.T)
    np.testing.assert_allclose(Vp, scale * A_inv, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(H_coef, A_inv @ XWX, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        Vf,
        scale * (A_inv_sym @ XWX @ A_inv_sym.T),
        atol=1e-12,
        rtol=0.0,
    )


def test_select_covariance_matrix_uses_model_default_and_override(monkeypatch):
    """
    Owner-contract coverage verifying that select covariance matrix uses model default
    and override.
    """
    model = SimpleNamespace(covariance="bayes")
    bayes = np.array([[1.0, 0.1], [0.1, 2.0]], dtype=np.float64)
    freq = np.array([[3.0, 0.2], [0.2, 4.0]], dtype=np.float64)

    monkeypatch.setattr(covariance_module, "_cov_bayes", lambda obj: bayes)
    monkeypatch.setattr(covariance_module, "_cov_freq", lambda obj: freq)

    np.testing.assert_allclose(select_covariance_matrix(model), bayes)
    np.testing.assert_allclose(select_covariance_matrix(model, cov="freq"), freq)
    np.testing.assert_allclose(select_covariance_matrix(model, cov=freq), freq)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript required for mgcv parity")
def test_gaussian_exact_covariance_assembly_matches_mgcv_snapshot():
    """
    Owner-contract coverage verifying that gaussian exact covariance assembly matches
    mgcv snapshot.
    """
    case = get_parity_case("gaussian_cr_uni_reml")
    data = make_parity_case_data(case.case_id)
    formula = case.formula

    expected = _run_mgcv_snapshot(data, formula, case.family, case.method)
    gam = _fit_nampy_model(data, formula, case.family, case.method)
    fit_result = gam.gam_result_.fit_core_solution.fit_result

    np.testing.assert_allclose(
        np.asarray(fit_result.cov_bayes, dtype=np.float64),
        np.asarray(expected["fit"]["cov_bayes"], dtype=np.float64),
        atol=1e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(fit_result.cov_freq, dtype=np.float64),
        np.asarray(expected["fit"]["cov_freq"], dtype=np.float64),
        atol=1e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        select_covariance_matrix(gam, cov="bayes"),
        np.asarray(expected["fit"]["cov_bayes"], dtype=np.float64),
        atol=1e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        select_covariance_matrix(gam, cov="freq"),
        np.asarray(expected["fit"]["cov_freq"], dtype=np.float64),
        atol=1e-6,
        rtol=0.0,
    )


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript required for mgcv parity")
def test_gaussian_stacked_qr_covariance_assembly_matches_mgcv_snapshot():
    """
    Owner-contract coverage verifying that gaussian stacked QR covariance assembly
    matches mgcv snapshot.
    """
    case = get_parity_case("gaussian_re_reml")
    data = make_parity_case_data(case.case_id)
    formula = case.formula

    expected = _run_mgcv_snapshot(data, formula, case.family, case.method)
    gam = _fit_nampy_model(data, formula, case.family, case.method)
    fit_result = gam.gam_result_.fit_core_solution.fit_result

    np.testing.assert_allclose(
        np.asarray(fit_result.cov_bayes, dtype=np.float64),
        np.asarray(expected["fit"]["cov_bayes"], dtype=np.float64),
        atol=1e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(fit_result.cov_freq, dtype=np.float64),
        np.asarray(expected["fit"]["cov_freq"], dtype=np.float64),
        atol=1e-6,
        rtol=0.0,
    )
    assert float(fit_result.trace_H) == pytest.approx(
        float(expected["fit"]["trace_H"]),
        abs=1e-7,
    )
    assert float(fit_result.scale) == pytest.approx(
        float(expected["fit"]["scale"]),
        abs=1e-8,
    )


def test_covariance_constructor_kwarg_switches_default_se_surface():
    """GAM(covariance="freq") changes every default SE surface consistently.

    The constructor kwarg was previously never passed by any test even though
    it silently switches `_select_cov(None)` — and with it every
    `predict(return_se=True)` — from the Bayesian Vp to the frequentist Ve.
    The freq-model default SEs must equal the bayes-model SEs computed with an
    explicit `cov="freq"` selection, and genuinely differ from the Vp SEs.
    """
    case = get_parity_case("gaussian_cr_uni_reml")
    data = make_parity_case_data(case.case_id)

    gam_bayes = _fit_nampy_model(data, case.formula, case.family, case.method)
    gam_freq = GAM(
        family=case.family,
        formula=case.formula,
        optimize_smoothing=True,
        smoothing_method=case.method,
        covariance="freq",
    ).fit(data=data)

    np.testing.assert_allclose(
        select_covariance_matrix(gam_freq, cov=None),
        select_covariance_matrix(gam_bayes, cov="freq"),
        atol=1e-12,
        rtol=0.0,
    )

    _pred_b, se_default_bayes = gam_bayes.predict(data, return_se=True)
    _pred_f, se_default_freq = gam_freq.predict(data, return_se=True)
    _pred_e, se_explicit_freq = gam_bayes.predict(data, return_se=True, cov="freq")

    np.testing.assert_allclose(
        np.asarray(se_default_freq, dtype=np.float64),
        np.asarray(se_explicit_freq, dtype=np.float64),
        atol=1e-12,
        rtol=0.0,
    )
    assert not np.allclose(
        np.asarray(se_default_freq, dtype=np.float64),
        np.asarray(se_default_bayes, dtype=np.float64),
        atol=1e-10,
        rtol=1e-10,
    )
