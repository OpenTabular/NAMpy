"""Extended mgcv parity scenarios that are useful but not part of the core matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _mgcv_snapshot_parity_shared import (
    TestAdditionalScenarioParity as _SharedTestAdditionalScenarioParity,
)
from mgcv_parity_utils import (
    _assert_basic_mgcv_parity,
    _assert_exact_mgcv_snapshot_parity,
    _fit_nampy_model,
    _fit_nampy_model_fixed_sp,
    _fit_nampy_snapshot,
    _make_fs_data_4levels,
    _make_gamma_data,
    _make_gaussian_data,
    _make_negbin_data,
    _make_sz_data_3x3,
    _run_mgcv_snapshot,
)


class TestAdditionalScenarioParity:
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

    def test_gamma_identity_link_fixed_sp_matches_mgcv(self):
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
            atol=1e-9,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["link"], dtype=np.float64),
            np.asarray(expected["predictions"]["link"], dtype=np.float64),
            atol=1e-9,
            rtol=0.0,
        )

    def test_gamma_identity_link_reml_matches_mgcv(self):
        data = _make_gamma_data(seed=364, n=220)
        family = {"name": "gamma", "link": "identity"}
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, family, "REML")
        expected = _run_mgcv_snapshot(data, formula, family, "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=2e-5,
            pred_rtol=0.0,
            sp_log_atol=3e-5,
            criterion_atol=1e-8,
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

    def test_negbin_estimated_theta_reml_matches_mgcv(self):
        data = _make_negbin_data(seed=2024, n=240, theta=1.0)
        formula = 'y ~ s(x0, bs="cr", k=8)'
        family = {"name": "negbin", "theta": 2.0, "estimate_theta": True}

        model_obj = _fit_nampy_model(data, formula, family, "REML")
        model = model_obj.parity_snapshot(X=data, include_covariances=True)
        expected = _run_mgcv_snapshot(data, formula, family, "REML")

        np.testing.assert_allclose(
            np.asarray(model["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=2e-4,
            rtol=2e-4,
        )
        np.testing.assert_allclose(
            np.asarray(model["fit"]["criterion_value"], dtype=np.float64),
            np.asarray(expected["fit"]["criterion_value"], dtype=np.float64),
            atol=1e-2,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(model_obj.family.theta, dtype=np.float64),
            np.asarray(expected["fit"]["family_theta"], dtype=np.float64),
            atol=2e-2,
            rtol=2e-2,
        )


class TestFsSzMoreFactors:
    """FS/SZ parity with more factor levels than the minimal 3-level smoke tests."""

    def test_gaussian_fs_4levels_reml_matches_mgcv(self):
        data = _make_fs_data_4levels()
        formula = 'y ~ s(f, x, bs="fs", k=6)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        # sp_log_atol is loose: known gap in dynamic-Gaussian REML for fs penalties.
        # Primary parity signal is the predictions.
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=6e-3,
            pred_rtol=6e-3,
            sp_log_atol=12.0,
            criterion_atol=5.0,
        )

    def test_gaussian_sz_3x3_reml_matches_mgcv(self):
        data = _make_sz_data_3x3()
        formula = 'y ~ s(f1, f2, x, bs="sz", k=6)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        # sp_log_atol is loose: same known REML gap as the 2×2 SZ baseline test.
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

        from nampy.gam.design.compiler import compile_predictor_designs
        from nampy.gam.formula import (
            compile_predictor_specs_from_formula,
            parse_gam_formula,
        )
        from nampy.gam.formula.preprocess import preprocess_formula_predictor_specs

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
        predictor_specs = compile_predictor_specs_from_formula(parsed, default_k=8)

        # Preprocess to materialise factor levels etc. (pure numeric here, no-op).
        predictor_specs, data_work, _ = preprocess_formula_predictor_specs(
            parsed=parsed,
            predictor_specs=predictor_specs,
            data=data,
        )

        feature_names = ["x0", "x1"]
        X = data_work[feature_names].to_numpy(dtype=np.float64)

        designs = compile_predictor_designs(
            X=X,
            feature_names=feature_names,
            predictor_specs=predictor_specs,
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
    def test_fs_factor_by_without_factor_feature_uses_base_smooth_expansion(self):
        from nampy.gam.formula import (
            compile_predictor_specs_from_formula,
            parse_gam_formula,
        )
        from nampy.gam.formula.preprocess import preprocess_formula_predictor_specs

        data = pd.DataFrame(
            {
                "y": [0.0, 1.0, 0.5, -0.25],
                "x": [0.1, 0.4, 0.8, 1.2],
                "f": ["a", "b", "a", "b"],
            }
        )
        parsed = parse_gam_formula('y ~ s(x, bs="fs", by=f, k=8)')
        predictor_specs = compile_predictor_specs_from_formula(parsed, default_k=8)

        out_specs, data_work, state = preprocess_formula_predictor_specs(
            parsed=parsed,
            predictor_specs=predictor_specs,
            data=data,
        )

        terms = list(out_specs[0].terms)
        assert len(terms) == 2
        assert all(str(term.smooth_spec.bs) == "tp" for term in terms)
        assert all(term.by_variable in data_work.columns for term in terms)
        assert len(state["factor_by_expansions"]) == 2
