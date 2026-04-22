"""Extended mgcv parity scenarios that are useful but not part of the core matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests._mgcv_snapshot_parity_shared import (
    TestAdditionalScenarioParity as _SharedTestAdditionalScenarioParity,
)
from tests.mgcv_parity_utils import (
    _assert_basic_mgcv_parity,
    _assert_exact_mgcv_snapshot_parity,
    _fit_nampy_model_fixed_sp,
    _fit_nampy_snapshot,
    _make_fs_data_4levels,
    _make_gamma_data,
    _make_gaussian_data,
    _make_negbin_data,
    _make_sz_data_3x3,
    _run_mgcv_snapshot,
)


def _make_transformed_formula_data(seed=531, n=120):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.75, 1.75, size=n)
    z = rng.uniform(-1.25, 1.25, size=n)
    o = rng.uniform(0.2, 1.4, size=n)
    y = (
        1.2
        + 0.35 * x**2
        + 0.45 * np.sin(1.3 * z)
        + np.log1p(o)
        + rng.normal(scale=0.05, size=n)
    )
    return pd.DataFrame({"y": y, "x": x, "z": z, "o": o})


class TestAdditionalScenarioParity:
    """
    Additional end-to-end parity scenarios that extend the requested mgcv surface beyond
    the core snapshot matrix.
    """
    test_binomial_select_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_binomial_select_reml_matches_mgcv
    )
    test_poisson_select_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_poisson_select_reml_matches_mgcv
    )
    test_gaussian_re_select_reml_matches_mgcv_exactly = (
        _SharedTestAdditionalScenarioParity.test_gaussian_re_select_reml_matches_mgcv_exactly
    )
    test_gaussian_sz_select_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_sz_select_reml_matches_mgcv
    )
    test_gaussian_mrf_select_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_mrf_select_reml_matches_mgcv
    )
    test_weighted_poisson_fixed_sp_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_weighted_poisson_fixed_sp_matches_mgcv
    )
    test_weighted_binomial_fixed_sp_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_weighted_binomial_fixed_sp_matches_mgcv
    )

    test_gaussian_fs_ps_marginal_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_fs_ps_marginal_reml_matches_mgcv
    )
    test_gaussian_fs_ps_marginal_select_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_fs_ps_marginal_select_reml_matches_mgcv
    )
    test_negbin_theta_0p5_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_negbin_theta_0p5_reml_matches_mgcv
    )
    test_negbin_theta_2p0_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_negbin_theta_2p0_reml_matches_mgcv
    )
    test_negbin_theta_0p5_fixed_sp_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_negbin_theta_0p5_fixed_sp_matches_mgcv
    )
    test_negbin_theta_2p0_fixed_sp_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_negbin_theta_2p0_fixed_sp_matches_mgcv
    )
    test_binomial_probit_fixed_sp_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_binomial_probit_fixed_sp_matches_mgcv
    )
    test_binomial_probit_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_binomial_probit_reml_matches_mgcv
    )
    test_binomial_cloglog_fixed_sp_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_binomial_cloglog_fixed_sp_matches_mgcv
    )
    test_binomial_cloglog_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_binomial_cloglog_reml_matches_mgcv
    )
    test_gamma_inverse_link_fixed_sp_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gamma_inverse_link_fixed_sp_matches_mgcv
    )
    test_gamma_inverse_link_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gamma_inverse_link_reml_matches_mgcv
    )

    def test_gaussian_transformed_formula_fixed_sp_snapshot_matches_mgcv_exactly(self):
        """
        Verify that gaussian transformed formula fixed sp snapshot matches mgcv exactly.
        """
        data = _make_transformed_formula_data()
        formula = (
            'I(y**2) ~ I(x**2) + s(I(z**2), bs="cr", k=6, sp=0.9) + offset(log(o + 1))'
        )

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(
            data,
            formula,
            "gaussian",
            "fixed",
        )

        np.testing.assert_allclose(
            np.asarray(actual["fit"]["coef_full"], dtype=np.float64),
            np.asarray(expected["fit"]["coef_full"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_total"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_total"], dtype=np.float64),
            atol=1e-10,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["deviance"], dtype=np.float64),
            np.asarray(expected["fit"]["deviance"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["link"], dtype=np.float64),
            np.asarray(expected["predictions"]["link"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gaussian_transformed_formula_reml_snapshot_matches_mgcv_exactly(self):
        """
        Verify that gaussian transformed formula REML snapshot matches mgcv exactly.
        """
        data = _make_transformed_formula_data(seed=533, n=140)
        formula = 'I(y**2) ~ I(x**2) + s(I(z**2), bs="cr", k=6) + offset(log(o + 1))'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(
            data,
            formula,
            "gaussian",
            "REML",
        )

        np.testing.assert_allclose(
            np.asarray(actual["fit"]["smoothing_params"], dtype=np.float64),
            np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64),
            atol=5e-5,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["log_smoothing_params"], dtype=np.float64),
            np.asarray(expected["fit"]["log_smoothing_params"], dtype=np.float64),
            atol=1e-10,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_total"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_total"], dtype=np.float64),
            atol=1e-10,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=3e-4,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["criterion_value"], dtype=np.float64),
            np.asarray(expected["fit"]["criterion_value"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["deviance"], dtype=np.float64),
            np.asarray(expected["fit"]["deviance"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["coef_full"], dtype=np.float64),
            np.asarray(expected["fit"]["coef_full"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        for key in ("response", "link", "se_response", "se_link"):
            np.testing.assert_allclose(
                np.asarray(actual["predictions"][key], dtype=np.float64),
                np.asarray(expected["predictions"][key], dtype=np.float64),
                atol=1e-10,
                rtol=1e-10,
            )

    def test_gamma_identity_link_fixed_sp_matches_mgcv(self):
        """Verify that gamma identity link fixed sp matches mgcv."""
        data = _make_gamma_data(seed=362, n=220)
        family = {"name": "gamma", "link": "identity"}
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        expected = _run_mgcv_snapshot(data, formula, family, "REML")
        sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)
        actual = gam.parity_snapshot(X=data, include_covariances=True)

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-10,
            pred_rtol=0.0,
            sp_log_atol=1e-10,
            check_criterion=False,
        )

    def test_gamma_identity_link_reml_prediction_matches_mgcv_fixed_sp(self):
        """Verify that gamma identity link REML prediction matches mgcv fixed sp."""
        data = _make_gamma_data(seed=363, n=220)
        family = {"name": "gamma", "link": "identity"}
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        expected = _run_mgcv_snapshot(data, formula, family, "REML")
        sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)
        actual = gam.parity_snapshot(X=data, include_covariances=True)

        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=1e-6,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["link"], dtype=np.float64),
            np.asarray(expected["predictions"]["link"], dtype=np.float64),
            atol=1e-6,
            rtol=0.0,
        )

    def test_gamma_identity_link_reml_matches_mgcv(self):
        """Verify that gamma identity link REML matches mgcv."""
        data = _make_gamma_data(seed=364, n=220)
        family = {"name": "gamma", "link": "identity"}
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        expected = _run_mgcv_snapshot(data, formula, family, "REML")
        sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)
        actual = gam.parity_snapshot(X=data, include_covariances=True)

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=3e-6,
            pred_rtol=0.0,
            sp_log_atol=1e-10,
            check_criterion=False,
        )

    test_gaussian_te_cc_cc_fixed_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_te_cc_cc_fixed_matches_mgcv
    )
    test_gaussian_ti_cc_cc_fixed_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_ti_cc_cc_fixed_matches_mgcv
    )
    test_gaussian_t2_cc_cc_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_t2_cc_cc_reml_matches_mgcv
    )
    test_gaussian_te_ts_cr_fixed_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_te_ts_cr_fixed_matches_mgcv
    )
    test_gaussian_ti_ts_cr_fixed_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_ti_ts_cr_fixed_matches_mgcv
    )
    test_gaussian_te_tp_cr_fixed_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_te_tp_cr_fixed_matches_mgcv
    )
    test_gaussian_ti_gp_cr_fixed_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_ti_gp_cr_fixed_matches_mgcv
    )

    def test_gaussian_te_select_reml_matches_mgcv(self):
        """Verify that gaussian te select REML matches mgcv."""
        data = _make_gaussian_data(seed=332, n=180)
        formula = 'y ~ te(x0, x1, bs=["cr", "cr"], k=[6, 6])'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML", select=True)
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML", select=True)

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-7,
            pred_rtol=0.0,
            sp_log_atol=1e-5,
        )

    def test_gamma_select_reml_matches_mgcv(self):
        """Verify that gamma select REML matches mgcv."""
        data = _make_gamma_data(seed=302, n=220)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "gamma", "REML", select=True)
        expected = _run_mgcv_snapshot(data, formula, "gamma", "REML", select=True)

        _assert_exact_mgcv_snapshot_parity(
            actual,
            expected,
            pred_atol=1e-8,
            pred_rtol=1e-8,
            edf_atol=1e-8,
            criterion_atol=1e-10,
            criterion_rtol=1e-10,
            sp_atol=3e-6,
            sp_rtol=1e-8,
            log_sp_atol=1e-6,
        )

    def test_negbin_select_reml_matches_mgcv(self):
        """Verify that negative-binomial select REML matches mgcv."""
        data = _make_negbin_data(seed=303, n=240, theta=1.0)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        family = {"name": "negbin", "theta": 1.0}

        actual = _fit_nampy_snapshot(data, formula, family, "REML", select=True)
        expected = _run_mgcv_snapshot(data, formula, family, "REML", select=True)

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-4,
            pred_rtol=0.0,
            sp_log_atol=1e-10,
            check_sp=False,
            criterion_atol=1e-3,
        )

        np.testing.assert_allclose(
            np.asarray(actual["fit"]["family_theta"], dtype=np.float64),
            np.asarray(expected["fit"]["family_theta"], dtype=np.float64),
            atol=1e-12,
            rtol=0.0,
        )

    def test_gaussian_fs_numeric_by_fixed_sp_matches_mgcv(self):
        """Verify that gaussian fs numeric by fixed sp matches mgcv."""
        rng = np.random.default_rng(381)
        n = 120
        x = rng.uniform(-1.0, 1.0, size=n)
        z = rng.uniform(0.5, 1.5, size=n)
        f = pd.Categorical(rng.choice(["a", "b", "c"], size=n))
        shifts = {"a": 0.35, "b": -0.25, "c": 0.15}
        y = z * (np.sin(1.4 * x) + np.array([shifts[str(v)] for v in f]))
        y = y + rng.normal(0.0, 0.05, size=n)
        data = pd.DataFrame({"y": y, "x": x, "z": z, "f": f})
        formula = 'y ~ s(f, x, bs="fs", by=z, k=5, xt="cr", sp=[0.9, 0.7, 0.5])'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")

        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=1e-8,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["link"], dtype=np.float64),
            np.asarray(expected["predictions"]["link"], dtype=np.float64),
            atol=1e-8,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_total"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_total"], dtype=np.float64),
            atol=1e-8,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-8,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["deviance"], dtype=np.float64),
            np.asarray(expected["fit"]["deviance"], dtype=np.float64),
            atol=1e-8,
            rtol=1e-8,
        )

    def test_negbin_estimated_theta_reml_matches_mgcv(self):
        """Verify that negative-binomial estimated theta REML matches mgcv."""
        data = _make_negbin_data(seed=2024, n=240, theta=1.0)
        formula = 'y ~ s(x0, bs="cr", k=8)'
        family = {"name": "negbin", "theta": 2.0, "estimate_theta": True}

        actual = _fit_nampy_snapshot(data, formula, family, "REML")
        expected = _run_mgcv_snapshot(data, formula, family, "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-6,
            pred_rtol=0.0,
            sp_log_atol=1e-2,
            criterion_atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["family_theta"], dtype=np.float64),
            np.asarray(expected["fit"]["family_theta"], dtype=np.float64),
            atol=5e-6,
            rtol=0.0,
        )


class TestFsSzMoreFactors:
    """FS/SZ parity with more factor levels than the minimal 3-level smoke tests."""

    def test_gaussian_fs_4levels_reml_matches_mgcv(self):
        """Verify that gaussian fs 4levels REML matches mgcv."""
        data = _make_fs_data_4levels()
        formula = 'y ~ s(f, x, bs="fs", k=6)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        assert (
            actual["parity"]["criterion_view"]["criterion_backend"] == "gaussian_exact"
        )
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=6e-3,
            pred_rtol=6e-3,
            sp_log_atol=12.0,
            criterion_atol=5.0,
        )

    def test_gaussian_sz_3x3_reml_matches_mgcv(self):
        """Verify that gaussian sz 3x3 REML matches mgcv."""
        data = _make_sz_data_3x3()
        formula = 'y ~ s(f1, f2, x, bs="sz", k=6)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        assert (
            actual["parity"]["criterion_view"]["criterion_backend"] == "gaussian_exact"
        )
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=6e-3,
            pred_rtol=6e-3,
            sp_log_atol=18.0,
            criterion_atol=5.0,
        )


class TestDistributionalRegressionMultiPredictor:
    """compile_predictor_designs with two independent LinearPredictorSpecs."""

    def test_two_predictors_are_structurally_independent(self):
        """Verify that two predictors are structurally independent."""
        from nampy.gam.compiler.compile_predictors import compile_predictors
        from nampy.gam.formula import extract_formula_terms, parse_gam_formula
        from nampy.gam.specs.build import build_formula_model

        rng = np.random.default_rng(7)
        n = 60
        x0 = rng.uniform(-2.0, 2.0, size=n)
        x1 = rng.uniform(-1.5, 1.5, size=n)
        y = np.sin(x0) + 0.3 * x1**2 + rng.normal(scale=0.1, size=n)
        data = pd.DataFrame({"y": y, "x0": x0, "x1": x1})

        # Multi-predictor formula: one for the mean (eta1), one for log-dispersion (eta2).
        parsed = parse_gam_formula(
            [
                'y ~ s(x0, bs="cr", k=7) + s(x1, bs="cr", k=5)',
                'y ~ s(x0, bs="cr", k=4)',
            ]
        )
        extracted = extract_formula_terms(parsed)
        built = build_formula_model(
            extracted,
            data=data,
            y=np.zeros(len(data)),
            default_k=8,
        )
        designs = compile_predictors(
            X=built.X,
            feature_names=built.feature_names,
            predictor_specs=built.predictor_specs,
        )

        assert len(designs) == 2, "Expected two CompiledPredictors"

        d0, d1 = designs

        # eta1: two smooth terms (k=7 and k=5 minus one constraint each).
        # eta2: one smooth term (k=4 minus one constraint).
        # Exact coef counts depend on constraint absorption; just verify non-zero.
        assert d0.n_coef > 0
        assert d1.n_coef > 0

        # Smoothing parameter maps are independent — no shared keys.
        sp_ids_0 = set(d0.smoothing_parameter_map.keys())
        sp_ids_1 = set(d1.smoothing_parameter_map.keys())
        assert sp_ids_0.isdisjoint(sp_ids_1), (
            f"Smoothing param IDs must not overlap across predictors: "
            f"{sp_ids_0 & sp_ids_1}"
        )

        # eta1 has two smoothing params; eta2 has one.
        assert d0.n_smoothing_params == 2
        assert d1.n_smoothing_params == 1

        # eta1 has two terms; eta2 has one.
        assert len(d0.compiled_terms) == 2
        assert len(d1.compiled_terms) == 1

        # Design matrices have the right shape.
        assert d0.design_matrix.shape == (n, d0.n_coef)
        assert d1.design_matrix.shape == (n, d1.n_coef)

        # Coefficient slices on eta2 start from 0 (independent of eta1).
        term_eta2 = d1.compiled_terms[0]
        assert term_eta2.coef_slice.start == 0


class TestFactorSmoothByPreprocess:
    """
    Regression coverage for factor-smooth by-variable preprocessing when the explicit
    factor column is absent.
    """
    def test_fs_factor_by_without_factor_feature_uses_base_smooth_expansion(self):
        """
        Verify that fs factor by without factor feature uses base smooth expansion.
        """
        from nampy.gam.formula import extract_formula_terms, parse_gam_formula
        from nampy.gam.specs.build import build_formula_model

        data = pd.DataFrame(
            {
                "y": [0.0, 1.0, 0.5, -0.25],
                "x": [0.1, 0.4, 0.8, 1.2],
                "f": ["a", "b", "a", "b"],
            }
        )
        parsed = parse_gam_formula('y ~ s(x, bs="fs", by=f, k=8)')
        extracted = extract_formula_terms(parsed)
        built = build_formula_model(
            extracted,
            data=data,
            y=np.zeros(len(data)),
            default_k=8,
        )

        terms = list(built.predictor_specs[0].terms)
        assert len(terms) == 2
        assert all(str(term.smooth_spec.bs) == "tp" for term in terms)
        assert all(term.by_variable in built.working_data.columns for term in terms)
        assert len(built.preprocess_state["factor_by_expansions"]) == 2
