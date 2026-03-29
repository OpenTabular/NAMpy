from __future__ import annotations

import re

import numpy as np

from mgcv_parity_utils import (
    _fit_nampy_model,
    _fit_nampy_snapshot,
    _make_gaussian_data,
    _make_negbin_data,
    _make_poisson_data,
    _run_mgcv_snapshot,
)


def _norm_labels(labels):
    vals = [labels] if isinstance(labels, str) else list(labels)
    out = []
    for lab in vals:
        s = str(lab)
        s = re.sub(r',\s*bs\s*=\s*(?:\"[^\"]*\"|\[[^\]]*\])', "", s)
        s = re.sub(r',\s*k\s*=\s*(?:[^,\)]+|\[[^\]]*\])', "", s)
        s = re.sub(r"\s+", "", s)
        s = s.replace("])", ")")
        s = re.sub(r",\d+\)$", ")", s)
        out.append(s)
    return out


def test_mgcv_concurvity_parity_gaussian_reml():
    data = _make_gaussian_data(seed=902, n=160)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    a_diag = actual["parity"]["diagnostics"]
    e_diag = expected["parity"]["diagnostics"]

    assert _norm_labels(a_diag["concurvity_labels"]) == _norm_labels(e_diag["concurvity_labels"])
    a_conc = np.asarray(a_diag["concurvity_full"], dtype=np.float64)
    e_conc = np.asarray(e_diag["concurvity_full"], dtype=np.float64)
    assert a_conc.shape == e_conc.shape == (3, 3)
    assert np.all(a_conc >= 0.0)
    assert np.all(e_conc >= 0.0)
    np.testing.assert_allclose(
        a_conc[0],
        e_conc[0],
        atol=0.03,
        rtol=0.0,
    )


def test_mgcv_smooth_covariance_and_edf1_parity_gaussian_reml():
    data = _make_gaussian_data(seed=902, n=160)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    a_diag = actual["parity"]["diagnostics"]
    e_diag = expected["parity"]["diagnostics"]

    assert _norm_labels(a_diag["smooth_cov_bayes"]["labels"]) == _norm_labels(e_diag["smooth_cov_bayes"]["labels"])
    assert _norm_labels(a_diag["smooth_cov_freq"]["labels"]) == _norm_labels(e_diag["smooth_cov_freq"]["labels"])
    assert _norm_labels(a_diag["smooth_edf1"]["labels"]) == _norm_labels(e_diag["smooth_edf1"]["labels"])

    a_edf1 = np.asarray(a_diag["smooth_edf1"]["values"], dtype=np.float64)
    e_edf1 = np.asarray(e_diag["smooth_edf1"]["values"], dtype=np.float64)
    np.testing.assert_allclose(a_edf1, e_edf1, atol=0.04, rtol=0.0)

    for got, want in zip(a_diag["smooth_cov_bayes"]["blocks"], e_diag["smooth_cov_bayes"]["blocks"]):
        got = np.asarray(got, dtype=np.float64)
        want = np.asarray(want, dtype=np.float64)
        assert got.shape == want.shape
        np.testing.assert_allclose(got, got.T, atol=1e-10, rtol=0.0)
        np.testing.assert_allclose(want, want.T, atol=1e-10, rtol=0.0)
        assert np.all(np.linalg.eigvalsh(got) >= -1e-10)
        assert np.all(np.linalg.eigvalsh(want) >= -1e-10)
        np.testing.assert_allclose(got, want, atol=0.13, rtol=0.0)

    for got, want in zip(a_diag["smooth_cov_freq"]["blocks"], e_diag["smooth_cov_freq"]["blocks"]):
        got = np.asarray(got, dtype=np.float64)
        want = np.asarray(want, dtype=np.float64)
        assert got.shape == want.shape
        np.testing.assert_allclose(got, got.T, atol=1e-10, rtol=0.0)
        np.testing.assert_allclose(want, want.T, atol=1e-10, rtol=0.0)
        assert np.all(np.linalg.eigvalsh(got) >= -1e-10)
        assert np.all(np.linalg.eigvalsh(want) >= -1e-10)
        np.testing.assert_allclose(got, want, atol=0.11, rtol=0.0)

    a_fun = a_diag["smooth_function_space"]
    e_fun = e_diag["smooth_function_space"]
    assert _norm_labels(a_fun["labels"]) == _norm_labels(e_fun["labels"])
    for got, want in zip(a_fun["fitted"], e_fun["fitted"]):
        np.testing.assert_allclose(
            np.asarray(got, dtype=np.float64),
            np.asarray(want, dtype=np.float64),
            atol=0.0045,
            rtol=0.0,
        )
    for got, want in zip(a_fun["variance_diag"], e_fun["variance_diag"]):
        np.testing.assert_allclose(
            np.asarray(got, dtype=np.float64),
            np.asarray(want, dtype=np.float64),
            atol=0.005,
            rtol=0.0,
        )


def test_mgcv_smooth_function_space_parity_with_parametric_prefix():
    data = _make_gaussian_data(seed=921, n=160)
    formula = "y ~ x0 + s(x1, k=8)"

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    a_fun = actual["parity"]["diagnostics"]["smooth_function_space"]
    e_fun = expected["parity"]["diagnostics"]["smooth_function_space"]
    e_labels = [e_fun["labels"]] if isinstance(e_fun["labels"], str) else list(e_fun["labels"])
    assert a_fun["labels"] == e_labels == ["s(x1)"]

    np.testing.assert_allclose(
        np.asarray(a_fun["fitted"][0], dtype=np.float64),
        np.asarray(e_fun["fitted"][0], dtype=np.float64),
        atol=0.006,
        rtol=0.0,
    )


def test_mgcv_postfit_diagnostics_parity_poisson_reml():
    data = _make_poisson_data(seed=903, n=220)
    formula = "y ~ s(x0, k=8) + s(x1, k=8)"

    actual = _fit_nampy_snapshot(data, formula, "poisson", "REML")
    expected = _run_mgcv_snapshot(data, formula, "poisson", "REML")

    a_diag = actual["parity"]["diagnostics"]
    e_diag = expected["parity"]["diagnostics"]

    a_sp_vcov = np.asarray(a_diag["sp_vcov"], dtype=np.float64)
    e_sp_vcov = np.asarray(e_diag["sp_vcov"], dtype=np.float64)
    assert a_sp_vcov.shape == e_sp_vcov.shape == (2, 2)
    np.testing.assert_allclose(a_sp_vcov, a_sp_vcov.T, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(e_sp_vcov, e_sp_vcov.T, atol=1e-10, rtol=0.0)
    assert np.all(np.diag(a_sp_vcov) > 0.0)
    assert np.all(np.diag(e_sp_vcov) > 0.0)
    np.testing.assert_allclose(a_sp_vcov, e_sp_vcov, atol=5e-4, rtol=1e-7)

    a_vcomp = np.asarray(a_diag["gam_vcomp"], dtype=np.float64)
    e_vcomp = np.asarray(e_diag["gam_vcomp"], dtype=np.float64)
    assert a_vcomp.shape == e_vcomp.shape == (2, 3)
    assert np.all(np.isfinite(a_vcomp))
    assert np.all(np.isfinite(e_vcomp))
    assert np.all(a_vcomp[:, 0] > 0.0)
    assert np.all(e_vcomp[:, 0] > 0.0)
    np.testing.assert_allclose(
        np.log(a_vcomp),
        np.log(e_vcomp),
        atol=5e-6,
        rtol=0.0,
    )

    a_one_se = np.asarray(a_diag["one_se_rule"], dtype=np.float64)
    e_one_se = np.asarray(e_diag["one_se_rule"], dtype=np.float64)
    assert a_one_se.shape == e_one_se.shape == (2,)
    assert np.all(a_one_se > 0.0)
    assert np.all(e_one_se > 0.0)
    np.testing.assert_allclose(
        np.log(a_one_se),
        np.log(e_one_se),
        atol=1e-6,
        rtol=0.0,
    )


def test_mgcv_negbin_tensor_te_ti_diagnostics_parity_reml():
    family = {"name": "negbin", "theta": 1.0}
    cases = [
        (_make_negbin_data(seed=61, n=240, theta=1.0), 'y ~ te(x0, x1, bs=["cr", "cr"], k=[6, 6])'),
        (_make_negbin_data(seed=63, n=240, theta=1.0), 'y ~ ti(x0, x1, bs=["cr", "cr"], k=[6, 6])'),
    ]

    for data, formula in cases:
        actual = _fit_nampy_snapshot(data, formula, family, "REML")
        expected = _run_mgcv_snapshot(data, formula, family, "REML")

        a_diag = actual["parity"]["diagnostics"]
        e_diag = expected["parity"]["diagnostics"]

        assert _norm_labels(a_diag["concurvity_labels"]) == _norm_labels(
            e_diag["concurvity_labels"]
        )
        np.testing.assert_allclose(
            np.asarray(a_diag["concurvity_full"], dtype=np.float64),
            np.asarray(e_diag["concurvity_full"], dtype=np.float64),
            atol=1e-8,
            rtol=0.0,
        )

        a_res = a_diag["residuals"]
        e_res = e_diag["residuals"]
        for key, atol in (
            ("response", 2e-8),
            ("pearson", 2e-8),
            ("scaled_pearson", 2e-8),
            ("deviance", 2e-8),
        ):
            np.testing.assert_allclose(
                np.asarray(a_res[key], dtype=np.float64),
                np.asarray(e_res[key], dtype=np.float64),
                atol=atol,
                rtol=0.0,
            )

        a_k = np.asarray(a_diag["k_check"]["values"], dtype=np.float64)
        e_k = np.asarray(e_diag["k_check"]["values"], dtype=np.float64)
        np.testing.assert_allclose(a_k[:, 0], e_k[:, 0], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(a_k[:, 1], e_k[:, 1], atol=0.15, rtol=0.0)
        np.testing.assert_allclose(a_k[:, 2], e_k[:, 2], atol=0.12, rtol=0.0)
        np.testing.assert_allclose(a_k[:, 3], e_k[:, 3], atol=0.26, rtol=0.0)

        a_sp_vcov = np.asarray(a_diag["sp_vcov"], dtype=np.float64)
        e_sp_vcov = np.asarray(e_diag["sp_vcov"], dtype=np.float64)
        np.testing.assert_allclose(a_sp_vcov, e_sp_vcov, atol=3e-2, rtol=1e-6)

        a_vcomp = np.asarray(a_diag["gam_vcomp"], dtype=np.float64)
        e_vcomp = np.asarray(e_diag["gam_vcomp"], dtype=np.float64)
        np.testing.assert_allclose(
            np.log(a_vcomp),
            np.log(e_vcomp),
            atol=5e-4,
            rtol=0.0,
        )

        a_one_se = np.asarray(a_diag["one_se_rule"], dtype=np.float64)
        e_one_se = np.asarray(e_diag["one_se_rule"], dtype=np.float64)
        np.testing.assert_allclose(
            np.log(a_one_se),
            np.log(e_one_se),
            atol=1e-5,
            rtol=0.0,
        )


def test_mgcv_negbin_tensor_t2_postfit_diagnostics_parity_reml():
    data = _make_negbin_data(seed=79, n=240, theta=1.0)
    family = {"name": "negbin", "theta": 1.0}
    formula = 'y ~ t2(x0, x1, bs=["cr", "cr"], k=[6, 6])'

    actual = _fit_nampy_snapshot(data, formula, family, "REML")
    expected = _run_mgcv_snapshot(data, formula, family, "REML")

    a_diag = actual["parity"]["diagnostics"]
    e_diag = expected["parity"]["diagnostics"]

    a_res = a_diag["residuals"]
    e_res = e_diag["residuals"]
    for key, atol in (
        ("response", 1e-10),
        ("pearson", 1e-10),
        ("scaled_pearson", 1e-10),
        ("deviance", 1e-10),
    ):
        np.testing.assert_allclose(
            np.asarray(a_res[key], dtype=np.float64),
            np.asarray(e_res[key], dtype=np.float64),
            atol=atol,
            rtol=0.0,
        )

    a_k = np.asarray(a_diag["k_check"]["values"], dtype=np.float64)
    e_k = np.asarray(e_diag["k_check"]["values"], dtype=np.float64)
    np.testing.assert_allclose(a_k[:, 0], e_k[:, 0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(a_k[:, 1], e_k[:, 1], atol=0.13, rtol=0.0)
    np.testing.assert_allclose(a_k[:, 2], e_k[:, 2], atol=0.12, rtol=0.0)
    np.testing.assert_allclose(a_k[:, 3], e_k[:, 3], atol=0.0, rtol=0.0)

    a_sp_vcov = np.asarray(a_diag["sp_vcov"], dtype=np.float64)
    e_sp_vcov = np.asarray(e_diag["sp_vcov"], dtype=np.float64)
    np.testing.assert_allclose(a_sp_vcov, e_sp_vcov, atol=2e-6, rtol=1e-6)

    a_vcomp = np.asarray(a_diag["gam_vcomp"], dtype=np.float64)
    e_vcomp = np.asarray(e_diag["gam_vcomp"], dtype=np.float64)
    np.testing.assert_allclose(
        np.log(a_vcomp),
        np.log(e_vcomp),
        atol=4e-7,
        rtol=0.0,
    )

    a_one_se = np.asarray(a_diag["one_se_rule"], dtype=np.float64)
    e_one_se = np.asarray(e_diag["one_se_rule"], dtype=np.float64)
    np.testing.assert_allclose(
        np.log(a_one_se),
        np.log(e_one_se),
        atol=2e-8,
        rtol=0.0,
    )


def test_mgcv_residuals_parity_gaussian_reml():
    data = _make_gaussian_data(seed=905, n=140)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    a_res = actual["parity"]["diagnostics"]["residuals"]
    e_res = expected["parity"]["diagnostics"]["residuals"]

    for key, atol in (
        ("response", 5e-7),
        ("working", 5e-7),
        ("pearson", 5e-7),
        ("scaled_pearson", 5e-7),
        ("deviance", 5e-7),
    ):
        np.testing.assert_allclose(
            np.asarray(a_res[key], dtype=np.float64),
            np.asarray(e_res[key], dtype=np.float64),
            atol=atol,
            rtol=0.0,
        )


def test_mgcv_k_check_parity_gaussian_reml():
    data = _make_gaussian_data(seed=904, n=120)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    a_k = actual["parity"]["diagnostics"]["k_check"]
    e_k = expected["parity"]["diagnostics"]["k_check"]

    assert _norm_labels(a_k["labels"]) == _norm_labels(e_k["labels"])
    a_vals = np.asarray(a_k["values"], dtype=np.float64)
    e_vals = np.asarray(e_k["values"], dtype=np.float64)
    assert a_vals.shape == e_vals.shape == (2, 4)
    np.testing.assert_allclose(a_vals[:, 0], e_vals[:, 0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(a_vals[:, 1], e_vals[:, 1], atol=0.05, rtol=0.0)
    np.testing.assert_allclose(a_vals[:, 2], e_vals[:, 2], atol=0.12, rtol=0.0)
    np.testing.assert_allclose(a_vals[:, 3], e_vals[:, 3], atol=0.26, rtol=0.0)


def test_mgcv_full_covariance_snapshot_parity_gaussian_reml():
    data = _make_gaussian_data(seed=906, n=140)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    model = _fit_nampy_model(data, formula, "gaussian", "REML")
    actual = model.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    a_vp = np.asarray(actual["fit"]["cov_bayes"], dtype=np.float64)
    e_vp = np.asarray(expected["fit"]["cov_bayes"], dtype=np.float64)
    a_vf = np.asarray(actual["fit"]["cov_freq"], dtype=np.float64)
    e_vf = np.asarray(expected["fit"]["cov_freq"], dtype=np.float64)

    assert a_vp.shape == e_vp.shape
    assert a_vf.shape == e_vf.shape
    np.testing.assert_allclose(a_vp, e_vp, atol=0.13, rtol=0.0)
    np.testing.assert_allclose(a_vf, e_vf, atol=0.11, rtol=0.0)


def test_k_check_and_gam_check_smoke():
    data = _make_gaussian_data(seed=904, n=120)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
    model = _fit_nampy_model(data, formula, "gaussian", "REML")

    k_table = model.k_check(subsample=120, n_rep=8, seed=0)
    assert list(k_table.columns) == ["k_prime", "edf", "k_index", "p_value"]
    assert len(k_table) == 2
    assert np.all(np.isfinite(k_table["k_prime"].to_numpy(dtype=np.float64)))
    assert np.all(np.isfinite(k_table["edf"].to_numpy(dtype=np.float64)))

    report = model.gam_check(type="deviance", k_sample=120, k_rep=8, seed=0)
    assert set(report["mgcv_comparable"]) == {"residual_type", "residuals", "k_check"}
    assert set(report["nampy_specific"]) == {"convergence", "model_rank"}
    assert report["mgcv_comparable"]["residual_type"] == "deviance"
    assert report["mgcv_comparable"]["k_check"] is not None
    assert report["mgcv_comparable"]["residuals"].shape == (len(data),)
    assert report["nampy_specific"]["convergence"]["method"] is not None
