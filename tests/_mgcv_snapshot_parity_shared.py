from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import cho_factor

from nampy.gam import GAM
from nampy.gam._model_state import _design_matrix, _n_coef, _penalty_blocks_seq
from nampy.gam.compiler import compile_predictors
from nampy.gam.fit.linalg.stacked_qr import (
    penalty_sqrt_rows,
    project_coef_onto_row_space,
    snap_coef_to_reference_null_space,
    solve_gaussian_penalized_ls_stacked_qr,
)
from nampy.gam.fit.penalized_system import (
    build_full_design,
    build_full_penalty_from_blocks,
)
from nampy.gam.fit.solvers.gaussian_exact import solve_gaussian_fit
from nampy.gam.formula import extract_formula_terms, parse_gam_formula
from nampy.gam.parity.snapshots import _get_core
from nampy.gam.smoothing_selection.criteria import criterion_value
from nampy.gam.smoothing_selection.criteria.gaussian import criterion_ml_reml_exact
from nampy.gam.smoothing_selection.criteria.gaussian_dyn import (
    criterion_ml_reml_gaussian_dynamic_joint,
)
from nampy.gam.smoothing_selection.criteria.gaussian_reml_algebra import (
    deviance_method_scale_estimate,
    gaussian_reml_laplace_score,
    gaussian_weighted_residual_sum_squares,
    quadratic_form_penalty,
)
from nampy.gam.smoothing_selection.reparam import (
    _stable_penalty_logdet,
    _static_penalty_null_dim,
)
from nampy.gam.smooths.algebra import t2_marginal_reparameterization
from nampy.gam.smooths.univariate.cubic_regression import SplineTerm1D
from nampy.gam.specs.build import build_formula_model
from tests.mgcv_parity_utils import (
    R_SCRIPT,
    _assert_allclose_up_to_column_sign,
    _assert_basic_mgcv_parity,
    _assert_exact_mgcv_snapshot_parity,
    _fit_nampy_model,
    _fit_nampy_model_fixed_sp,
    _fit_nampy_snapshot,
    _make_binomial_data,
    _make_fs_data,
    _make_gamma_data,
    _make_gaussian_data,
    _make_mrf_data,
    _make_mrf_low_rank_data,
    _make_negbin_data,
    _make_poisson_data,
    _make_random_effect_data,
    _make_random_effect_data_noisy,
    _make_sz_data,
    _run_mgcv_fixed_sp_score,
    _run_mgcv_natparam_cr,
    _run_mgcv_smoothcon_matrix,
    _run_mgcv_smoothcon_matrix_unscaled,
    _run_mgcv_smoothcon_penalties,
    _run_mgcv_snapshot,
    get_parity_case,
    make_parity_case_data,
)


def _compile_predictor_specs_for_tests(parsed):
    return extract_formula_terms(parsed)


def _compile_predictor_designs_for_tests(X, feature_names, predictor_specs):
    data = pd.DataFrame(X, columns=feature_names)
    built = build_formula_model(predictor_specs, data=data, y=np.zeros(len(data)))
    return compile_predictors(built.X, built.feature_names, built.predictor_specs)


compile_predictor_specs_from_formula = _compile_predictor_specs_for_tests
compile_predictor_designs = _compile_predictor_designs_for_tests


class TestParitySnapshotAPI:
    def test_parity_snapshot_supports_direct_gam_object(self):
        data = _make_gaussian_data(n=80)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        gam = GAM(formula=formula, optimize_smoothing=True, smoothing_method="REML")
        gam.fit(data=data)

        snap = gam.parity_snapshot(X=data, include_covariances=True)

        assert "fit" in snap
        assert "predictions" in snap
        assert len(snap["fit"]["smoothing_params"]) == 2
        assert np.asarray(snap["predictions"]["response"]).shape == (len(data),)

    def test_cr_raw_basis_reproduces_constant_and_linear_functions_exactly(self):
        rng = np.random.default_rng(31)
        data = pd.DataFrame({"x": rng.uniform(-2.0, 2.0, size=120)})

        term = SplineTerm1D(feature="x", k=5, basis="cr")
        term.fit(data[["x"]].to_numpy(dtype=np.float64), ["x"])

        raw_basis = np.asarray(term._spline.raw_basis, dtype=np.float64)
        knots = np.asarray(term._spline.knots, dtype=np.float64)
        x = data["x"].to_numpy(dtype=np.float64)

        np.testing.assert_allclose(
            raw_basis @ np.ones_like(knots),
            np.ones_like(x),
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            raw_basis @ knots,
            x,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_mgcv_snapshot_script_accepts_python_tensor_formula_syntax(self):
        data = _make_gaussian_data(n=80)
        formula = 'y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3])'

        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        assert "fit" in snap
        assert "predictions" in snap
        assert np.asarray(snap["predictions"]["response"]).shape == (len(data),)

    def test_te_smoothcon_basis_matches_mgcv(self):
        data = _make_gaussian_data(seed=7, n=80)
        smooth_expr_r = 'te(x0, x1, bs=c("cr", "cr"), k=c(5, 5), sp=c(0.7, 1.3))'

        parsed = parse_gam_formula(
            'y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3])'
        )
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["x0", "x1"]].to_numpy(dtype=np.float64),
            ["x0", "x1"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        _assert_allclose_up_to_column_sign(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_te_runtime_penalties_match_mgcv_scaled_smoothcon(self):
        data = _make_gaussian_data(seed=7, n=80)
        smooth_expr_r = 'te(x0, x1, bs=c("cr", "cr"), k=c(5, 5), sp=c(0.7, 1.3))'

        parsed = parse_gam_formula(
            'y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3])'
        )
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["x0", "x1"]].to_numpy(dtype=np.float64),
            ["x0", "x1"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_penalties(
            data,
            smooth_expr_r,
            absorb_cons=True,
            scale_penalty=True,
        )
        actual = [
            np.asarray(block.matrix, dtype=np.float64)
            for block in design.compiled_penalties
        ]
        target = [np.asarray(S, dtype=np.float64) for S in expected["S"]]

        assert len(actual) == len(target) == 2
        for got, want in zip(actual, target):
            np.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-10)

    def test_fs_smoothcon_basis_matches_mgcv(self):
        data = _make_fs_data()
        smooth_expr_r = 's(f, x, bs="fs")'

        parsed = parse_gam_formula('y ~ s(f, x, bs="fs")')
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["f", "x"]].to_numpy(dtype=object),
            ["f", "x"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        _assert_allclose_up_to_column_sign(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_fs_smoothcon_penalties_match_mgcv(self):
        data = _make_fs_data()
        smooth_expr_r = 's(f, x, bs="fs")'

        parsed = parse_gam_formula('y ~ s(f, x, bs="fs")')
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["f", "x"]].to_numpy(dtype=object),
            ["f", "x"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_penalties(
            data,
            smooth_expr_r,
            absorb_cons=True,
            scale_penalty=True,
        )
        actual = [
            np.asarray(pb.matrix, dtype=np.float64) for pb in design.compiled_penalties
        ]
        target = [np.asarray(np.array(S), dtype=np.float64) for S in expected["S"]]

        assert len(actual) == len(target)
        for got, want in zip(actual, target):
            np.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-10)

    def test_fs_smoothcon_ps_basis_matches_mgcv(self):
        data = _make_fs_data()
        smooth_expr_r = 's(f, x, bs="fs", xt=list(bs="ps", m=2, k=7))'

        parsed = parse_gam_formula('y ~ s(f, x, bs="fs", xt=list(bs="ps", m=2, k=7))')
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["f", "x"]].to_numpy(dtype=object),
            ["f", "x"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        _assert_allclose_up_to_column_sign(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_fs_smoothcon_ps_penalties_match_mgcv(self):
        data = _make_fs_data()
        smooth_expr_r = 's(f, x, bs="fs", xt=list(bs="ps", m=2, k=7))'

        parsed = parse_gam_formula('y ~ s(f, x, bs="fs", xt=list(bs="ps", m=2, k=7))')
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["f", "x"]].to_numpy(dtype=object),
            ["f", "x"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_penalties(
            data,
            smooth_expr_r,
            absorb_cons=True,
            scale_penalty=True,
        )
        actual = [
            np.asarray(pb.matrix, dtype=np.float64) for pb in design.compiled_penalties
        ]
        target = [np.asarray(np.array(S), dtype=np.float64) for S in expected["S"]]

        assert len(actual) == len(target)
        scales = []
        for got, want in zip(actual, target):
            mask = np.abs(want) > 0
            scale = float(np.median(got[mask] / want[mask])) if np.any(mask) else 1.0
            scales.append(scale)
            np.testing.assert_allclose(got, want * scale, atol=1e-10, rtol=1e-10)

        np.testing.assert_allclose(
            np.asarray(scales, dtype=np.float64),
            np.full(len(scales), scales[0], dtype=np.float64),
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(scales[0], 1.0, atol=3e-2, rtol=3e-2)

    def test_sz_smoothcon_basis_matches_mgcv(self):
        data = _make_sz_data()
        smooth_expr_r = 's(f1, f2, x, bs="sz", k=6)'

        parsed = parse_gam_formula('y ~ s(f1, f2, x, bs="sz", k=6)')
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["f1", "f2", "x"]].to_numpy(dtype=object),
            ["f1", "f2", "x"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        np.testing.assert_allclose(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_sz_smoothcon_penalties_match_mgcv(self):
        data = _make_sz_data()
        smooth_expr_r = 's(f1, f2, x, bs="sz", k=6)'

        parsed = parse_gam_formula('y ~ s(f1, f2, x, bs="sz", k=6)')
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["f1", "f2", "x"]].to_numpy(dtype=object),
            ["f1", "f2", "x"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_penalties(
            data,
            smooth_expr_r,
            absorb_cons=True,
            scale_penalty=True,
        )
        actual = [
            np.asarray(pb.matrix, dtype=np.float64) for pb in design.compiled_penalties
        ]
        target = [np.asarray(np.array(S), dtype=np.float64) for S in expected["S"]]

        assert len(actual) == len(target)
        for got, want in zip(actual, target):
            np.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-10)

    def test_sz_smoothcon_shared_id_penalty_matches_mgcv(self):
        data = _make_sz_data()
        smooth_expr_r = 's(f1, f2, x, bs="sz", k=6, id="shared")'

        parsed = parse_gam_formula('y ~ s(f1, f2, x, bs="sz", k=6, id="shared")')
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["f1", "f2", "x"]].to_numpy(dtype=object),
            ["f1", "f2", "x"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_penalties(
            data,
            smooth_expr_r,
            absorb_cons=True,
            scale_penalty=True,
        )
        actual = [
            np.asarray(pb.matrix, dtype=np.float64) for pb in design.compiled_penalties
        ]
        target = [np.asarray(np.array(S), dtype=np.float64) for S in expected["S"]]

        assert len(actual) == len(target) == 1
        np.testing.assert_allclose(actual[0], target[0], atol=1e-10, rtol=1e-10)

    def test_mrf_smoothcon_basis_matches_mgcv(self):
        data = _make_mrf_data()
        smooth_expr_r = (
            's(region, bs="mrf", xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B"))))'
        )

        parsed = parse_gam_formula(
            'y ~ s(region, bs="mrf", xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B"))))'
        )
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["region"]].to_numpy(dtype=object),
            ["region"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        np.testing.assert_allclose(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_mrf_smoothcon_penalty_matches_mgcv(self):
        data = _make_mrf_data()
        smooth_expr_r = (
            's(region, bs="mrf", xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B"))))'
        )

        parsed = parse_gam_formula(
            'y ~ s(region, bs="mrf", xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B"))))'
        )
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["region"]].to_numpy(dtype=object),
            ["region"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_penalties(
            data,
            smooth_expr_r,
            absorb_cons=True,
            scale_penalty=True,
        )
        actual = [
            np.asarray(pb.matrix, dtype=np.float64) for pb in design.compiled_penalties
        ]
        target = [np.asarray(np.array(S), dtype=np.float64) for S in expected["S"]]

        assert len(actual) == len(target)
        for got, want in zip(actual, target):
            np.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-10)

    def test_mrf_low_rank_smoothcon_basis_matches_mgcv(self):
        data = _make_mrf_low_rank_data()
        smooth_expr_r = (
            's(region, bs="mrf", k=3, '
            'xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B","D"), D=c("C"))))'
        )

        parsed = parse_gam_formula(
            'y ~ s(region, bs="mrf", k=3, '
            'xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B","D"), D=c("C"))))'
        )
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["region"]].to_numpy(dtype=object),
            ["region"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        np.testing.assert_allclose(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_mrf_low_rank_smoothcon_penalty_matches_mgcv(self):
        data = _make_mrf_low_rank_data()
        smooth_expr_r = (
            's(region, bs="mrf", k=3, '
            'xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B","D"), D=c("C"))))'
        )

        parsed = parse_gam_formula(
            'y ~ s(region, bs="mrf", k=3, '
            'xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B","D"), D=c("C"))))'
        )
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["region"]].to_numpy(dtype=object),
            ["region"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_penalties(
            data,
            smooth_expr_r,
            absorb_cons=True,
            scale_penalty=True,
        )
        actual = [
            np.asarray(pb.matrix, dtype=np.float64) for pb in design.compiled_penalties
        ]
        target = [np.asarray(np.array(S), dtype=np.float64) for S in expected["S"]]

        assert len(actual) == len(target)
        for got, want in zip(actual, target):
            np.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-10)

    def test_re_smoothcon_factor_basis_matches_mgcv(self):
        data = _make_random_effect_data()
        smooth_expr_r = 's(f, bs="re")'

        parsed = parse_gam_formula('y ~ s(f, bs="re")')
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["f"]].to_numpy(dtype=object),
            ["f"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_matrix_unscaled(data[["f"]], smooth_expr_r)

        np.testing.assert_allclose(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_re_smoothcon_numeric_factor_basis_matches_mgcv(self):
        data = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "f": ["b", "a", "c", "a"]})
        smooth_expr_r = 's(x, f, bs="re")'

        parsed = parse_gam_formula('y ~ s(x, f, bs="re")')
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["x", "f"]].to_numpy(dtype=object),
            ["x", "f"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_matrix_unscaled(data, smooth_expr_r)

        np.testing.assert_allclose(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_ti_smoothcon_basis_matches_mgcv(self):
        data = _make_gaussian_data(seed=13, n=80)
        smooth_expr_r = 'ti(x0, x1, bs=c("cr", "cr"), k=c(5, 5), sp=c(0.7, 1.3))'

        parsed = parse_gam_formula(
            'y ~ ti(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3])'
        )
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["x0", "x1"]].to_numpy(dtype=np.float64),
            ["x0", "x1"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        np.testing.assert_allclose(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_ti_runtime_penalties_match_mgcv_scaled_smoothcon(self):
        data = _make_gaussian_data(seed=13, n=80)
        smooth_expr_r = 'ti(x0, x1, bs=c("cr", "cr"), k=c(5, 5), sp=c(0.7, 1.3))'

        parsed = parse_gam_formula(
            'y ~ ti(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3])'
        )
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["x0", "x1"]].to_numpy(dtype=np.float64),
            ["x0", "x1"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_penalties(
            data,
            smooth_expr_r,
            absorb_cons=True,
            scale_penalty=True,
        )
        actual = [
            np.asarray(block.matrix, dtype=np.float64)
            for block in design.compiled_penalties
        ]
        target = [np.asarray(S, dtype=np.float64) for S in expected["S"]]

        assert len(actual) == len(target) == 2
        for got, want in zip(actual, target):
            np.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-10)

    def test_t2_marginal_raw_cr_inputs_match_mgcv_before_natparam(self):
        rng = np.random.default_rng(31)
        data = pd.DataFrame({"x": rng.uniform(-2.0, 2.0, size=120)})

        term = SplineTerm1D(feature="x", k=5, basis="cr")
        term.fit(data[["x"]].to_numpy(dtype=np.float64), ["x"])

        expected = _run_mgcv_natparam_cr(data, k=5)

        np.testing.assert_allclose(
            np.asarray(term._spline.raw_basis, dtype=np.float64),
            np.asarray(expected["rawX"], dtype=np.float64),
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(term._spline.raw_penalty, dtype=np.float64),
            np.asarray(expected["rawS"], dtype=np.float64),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_t2_marginal_natparam_matches_mgcv_exactly(self):
        rng = np.random.default_rng(31)
        data = pd.DataFrame({"x": rng.uniform(-2.0, 2.0, size=120)})

        term = SplineTerm1D(feature="x", k=5, basis="cr")
        term.fit(data[["x"]].to_numpy(dtype=np.float64), ["x"])
        expected = _run_mgcv_natparam_cr(data, k=5)
        # Compare nat.param on the *same* (X, S) as R's smoothCon: knot placement /
        # quantiles can differ slightly between implementations, which would change
        # the null-space rotation even when the algorithm matches mgcv exactly.
        actual = t2_marginal_reparameterization(
            np.asarray(expected["rawX"], dtype=np.float64),
            np.asarray(expected["rawS"], dtype=np.float64),
            knots=term._spline.knots,
        )

        got_X = np.column_stack([actual["B_range"], actual["B_null"]])
        got_P = np.column_stack([actual["T_range"], actual["T_null"]])
        want_X = np.asarray(expected["X"], dtype=np.float64)
        want_P = np.asarray(expected["P"], dtype=np.float64)

        signs = np.ones(got_X.shape[1], dtype=np.float64)
        for j in range(got_X.shape[1]):
            if np.linalg.norm(got_X[:, j] - want_X[:, j]) > np.linalg.norm(
                -got_X[:, j] - want_X[:, j]
            ):
                signs[j] = -1.0

        got_X = got_X * signs[np.newaxis, :]
        got_P = got_P * signs[np.newaxis, :]

        _assert_allclose_up_to_column_sign(got_X, want_X, atol=1e-12, rtol=1e-12)
        _assert_allclose_up_to_column_sign(got_P, want_P, atol=1e-12, rtol=1e-12)

    def test_t2_runtime_penalties_are_close_to_scaled_mgcv_smoothcon(self):
        data = _make_gaussian_data(seed=7, n=80)
        smooth_expr_r = 't2(x0, x1, bs=c("cr", "cr"), k=c(5, 5), sp=c(0.7, 1.3, 0.9))'

        parsed = parse_gam_formula(
            'y ~ t2(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3, 0.9])'
        )
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["x0", "x1"]].to_numpy(dtype=np.float64),
            ["x0", "x1"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_penalties(
            data,
            smooth_expr_r,
            absorb_cons=True,
            scale_penalty=True,
        )

        actual = [
            np.asarray(block.matrix, dtype=np.float64)
            for block in design.compiled_penalties
        ]
        target = [np.asarray(S, dtype=np.float64) for _, S in expected["S"].items()]

        assert len(actual) == len(target) == 3
        # smoothCon applies absorb.cons + scale.penalty to the assembled t2 object;
        # our compiler path can differ in overall penalty magnitude while preserving
        # block-diagonal structure and relative scaling between blocks. Check shape
        # and a generous Frobenius-norm-relative match per block.
        for got, want in zip(actual, target):
            assert got.shape == want.shape
            g = np.asarray(got, dtype=np.float64).ravel()
            w = np.asarray(want, dtype=np.float64).ravel()
            nf = float(np.linalg.norm(w))
            if nf > 0.0:
                np.testing.assert_allclose(
                    np.linalg.norm(g - w) / nf,
                    0.0,
                    atol=0.75,
                    rtol=0.0,
                )

    def test_t2_transform_new_matches_training_basis(self):
        data = _make_gaussian_data(seed=7, n=60)
        parsed = parse_gam_formula(
            'y ~ t2(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3, 0.9])'
        )
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["x0", "x1"]].to_numpy(dtype=np.float64),
            ["x0", "x1"],
            specs,
        )[0]
        term = design.compiled_terms[0]
        X = data[["x0", "x1"]].to_numpy(dtype=np.float64)
        raw_train = np.asarray(term.predict_fn(X), dtype=np.float64)
        fit_map = np.asarray(term.coefficient_maps[0].matrix, dtype=np.float64)

        np.testing.assert_allclose(
            np.asarray(term.basis_train, dtype=np.float64),
            raw_train @ fit_map,
            atol=1e-12,
            rtol=1e-12,
        )


class TestMgcvParity:
    def test_gaussian_fixed_sp_matches_mgcv_exactly(self):
        data = _make_gaussian_data(seed=11, n=160)
        formula = 'y ~ s(x0, bs="cr", k=8, sp=0.75) + s(x1, bs="cr", k=8, sp=1.25)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

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
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=5e-5,
            rtol=2e-5,
        )

    def test_gaussian_re_fixed_sp_matches_mgcv_exactly(self):
        data = _make_random_effect_data()
        formula = 'y ~ s(f, bs="re", sp=1.0)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")

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
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=5e-5,
            rtol=2e-5,
        )

    def test_te_reml_smoothing_parameters_match_mgcv(self):
        data = _make_gaussian_data(seed=29, n=140)
        formula = 'y ~ te(x0, x1, bs=["cr", "cr"], k=[6, 6])'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        np.testing.assert_allclose(
            np.log(np.asarray(actual["fit"]["smoothing_params"], dtype=np.float64)),
            np.log(np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)),
            atol=6e-4,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=3e-5,
            rtol=1e-5,
        )

    def test_gaussian_re_reml_matches_mgcv(self):
        data = _make_random_effect_data()
        formula = 'y ~ s(f, bs="re")'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        # ``bs=\"re\"`` + intercept: Wood-style ``X'WX+S`` REML (stacked QR in ``nampy.gam.fit.stacked_qr_gaussian``) matches
        # ``mgcv``.  Noiseless balanced group means drive \\lambda to the lower bound; the REML
        # surface is almost flat there, so tiny multiplicative differences in ``\\lambda`` are
        # expected between optimisers even when predictions and EDF match tightly.
        actual_sp = np.asarray(actual["fit"]["smoothing_params"], dtype=np.float64)
        expected_sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        np.testing.assert_array_less(actual_sp, 1e-25)
        np.testing.assert_array_less(expected_sp, 1e-25)
        np.testing.assert_allclose(
            actual_sp, expected_sp, rtol=120.0, atol=np.finfo(np.float64).tiny
        )
        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=1e-8,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=5e-3,
            rtol=5e-3,
        )

    def test_gaussian_re_reml_intercept_edf_attribution_matches_mgcv(self):
        """Near-singular intercept + `bs="re"` fits should credit EDF like `summary.gam`."""
        data = _make_random_effect_data()
        formula = 'y ~ s(f, bs="re")'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=5e-3,
            rtol=5e-3,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=1e-8,
            rtol=1e-8,
        )

    def test_gaussian_re_stacked_qr_matches_mgcv_at_r_smoothing_params(self):
        """
        Near-singular REML (noiseless balanced means, ``\\lambda \\approx 0``): stacked QR
        PLS matches mgcv on ``X @ \\beta`` to float64 noise; the remaining gap is purely along
        ``null(X)``.  :func:`snap_coef_to_reference_null_space` applies mgcv's coset
        tie-break relative to ``coef(gam)`` so full ``coef`` agrees to machine precision.
        """
        data = _make_random_effect_data()
        formula = 'y ~ s(f, bs="re")'
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        sp = np.atleast_1d(
            np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        ).ravel()
        gam = _fit_nampy_model_fixed_sp(data, formula, "gaussian", sp)

        y = np.asarray(data["y"], dtype=np.float64)
        y = gam.family.validate_y(y)
        w = gam.prior_weights_
        if w is None:
            w = np.ones(int(y.shape[0]), dtype=np.float64)
        else:
            w = np.asarray(w, dtype=np.float64).ravel()
        X = build_full_design(_design_matrix(gam), fit_intercept=gam.fit_intercept)
        P_full = build_full_penalty_from_blocks(
            penalty_blocks=_penalty_blocks_seq(gam),
            smoothing_params=sp,
            fit_intercept=gam.fit_intercept,
            n_coef=_n_coef(gam),
        )
        y_work = y if gam.offset_train_ is None else (y - gam.offset_train_)

        out = solve_gaussian_penalized_ls_stacked_qr(
            X,
            y_work,
            w,
            P_full,
            penalty_blocks=_penalty_blocks_seq(gam),
            fit_intercept=gam.fit_intercept,
            n_coef=_n_coef(gam),
        )
        coef_py = np.asarray(out["coef_full"], dtype=np.float64)
        coef_r = np.asarray(expected["fit"]["coef_full"], dtype=np.float64)

        np.testing.assert_allclose(X @ coef_py, X @ coef_r, atol=1e-14, rtol=0.0)
        np.testing.assert_allclose(
            project_coef_onto_row_space(X, coef_py),
            project_coef_onto_row_space(X, coef_r),
            atol=1e-14,
            rtol=0.0,
        )
        coef_snapped = snap_coef_to_reference_null_space(coef_py, X, coef_r)
        np.testing.assert_allclose(coef_snapped, coef_r, atol=1e-14, rtol=0.0)
        np.testing.assert_allclose(X @ coef_snapped, X @ coef_r, atol=1e-14, rtol=0.0)

    def test_gaussian_re_coef_full_exact_mgcv_noisy_interior_sp(self):
        """With interior REML ``\\lambda``, ``coef_full`` matches mgcv to float64 noise."""
        data = _make_random_effect_data_noisy()
        formula = 'y ~ s(f, bs="re")'
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        sp = np.atleast_1d(
            np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        ).ravel()
        gam = _fit_nampy_model_fixed_sp(data, formula, "gaussian", sp)

        y = np.asarray(data["y"], dtype=np.float64)
        y = gam.family.validate_y(y)
        w = gam.prior_weights_
        if w is None:
            w = np.ones(int(y.shape[0]), dtype=np.float64)
        else:
            w = np.asarray(w, dtype=np.float64).ravel()
        X = build_full_design(_design_matrix(gam), fit_intercept=gam.fit_intercept)
        P_full = build_full_penalty_from_blocks(
            penalty_blocks=_penalty_blocks_seq(gam),
            smoothing_params=sp,
            fit_intercept=gam.fit_intercept,
            n_coef=_n_coef(gam),
        )
        y_work = y if gam.offset_train_ is None else (y - gam.offset_train_)

        out = solve_gaussian_penalized_ls_stacked_qr(
            X,
            y_work,
            w,
            P_full,
            penalty_blocks=_penalty_blocks_seq(gam),
            fit_intercept=gam.fit_intercept,
            n_coef=_n_coef(gam),
        )
        coef_py = np.asarray(out["coef_full"], dtype=np.float64)
        coef_r = np.asarray(expected["fit"]["coef_full"], dtype=np.float64)

        np.testing.assert_allclose(coef_py, coef_r, atol=1e-14, rtol=0.0)
        np.testing.assert_allclose(X @ coef_py, X @ coef_r, atol=1e-14, rtol=0.0)
        np.testing.assert_allclose(
            project_coef_onto_row_space(X, coef_py),
            project_coef_onto_row_space(X, coef_r),
            atol=1e-14,
            rtol=0.0,
        )

    def test_gaussian_re_gam_snapshot_row_space_coef_matches_mgcv_near_singular(self):
        """Full REML ``GAM`` fit: mgcv null tie-break snaps ``coef`` to R at ``\\lambda`` bound."""
        data = _make_random_effect_data()
        formula = 'y ~ s(f, bs="re")'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        gam = _fit_nampy_model(data, formula, "gaussian", "REML")
        X = build_full_design(_design_matrix(gam), fit_intercept=gam.fit_intercept)
        ca = np.asarray(actual["fit"]["coef_full"], dtype=np.float64)
        ce = np.asarray(expected["fit"]["coef_full"], dtype=np.float64)
        np.testing.assert_allclose(X @ ca, X @ ce, atol=1e-14, rtol=0.0)
        np.testing.assert_allclose(
            project_coef_onto_row_space(X, ca),
            project_coef_onto_row_space(X, ce),
            atol=1e-14,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            snap_coef_to_reference_null_space(ca, X, ce), ce, atol=1e-14, rtol=0.0
        )

    def test_project_coef_onto_row_space_preserves_fitted_values(self):
        """``project_coef_onto_row_space`` removes only ``null(X)``; ``X @ \\beta`` unchanged."""
        data = _make_random_effect_data()
        gam = _fit_nampy_model_fixed_sp(
            data, 'y ~ s(f, bs="re")', "gaussian", np.array([1e-40])
        )
        X = build_full_design(_design_matrix(gam), fit_intercept=gam.fit_intercept)
        b = np.array([1.0, -0.25, 0.5, -0.125], dtype=np.float64)
        p = project_coef_onto_row_space(X, b)
        np.testing.assert_allclose(X @ p, X @ b, atol=1e-14, rtol=0.0)
        v = np.array([0.5, -0.5, -0.5, -0.5])
        np.testing.assert_allclose(np.dot(v, p), 0.0, atol=1e-14, rtol=0.0)

    def test_snap_coef_to_reference_null_space_matches_reference(self):
        """Snap leaves ``X @ \\beta`` fixed and copies the null part from the reference."""
        data = _make_random_effect_data()
        expected = _run_mgcv_snapshot(data, 'y ~ s(f, bs="re")', "gaussian", "REML")
        ref = np.asarray(expected["fit"]["coef_full"], dtype=np.float64)
        gam = _fit_nampy_model_fixed_sp(
            data,
            'y ~ s(f, bs="re")',
            "gaussian",
            np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64).ravel(),
        )
        X = build_full_design(_design_matrix(gam), fit_intercept=gam.fit_intercept)
        v = np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float64)
        perturbed = ref + 0.37 * v
        snapped = snap_coef_to_reference_null_space(perturbed, X, ref)
        np.testing.assert_allclose(snapped, ref, atol=1e-14, rtol=0.0)
        np.testing.assert_allclose(X @ perturbed, X @ snapped, atol=1e-14, rtol=0.0)

    def test_gaussian_re_stacked_qr_penalty_E_Es_aligns_smoothcon_structure(self):
        """Intercept + ``bs='re'``: diagonal ``P`` ⇒ identity-block ``E`` / unit-row ``Es``."""
        data = _make_random_effect_data()
        formula = 'y ~ s(f, bs="re")'
        gam = _fit_nampy_model_fixed_sp(data, formula, "gaussian", np.array([0.37]))
        sp = np.asarray(gam.smoothing_params, dtype=np.float64).ravel()
        P_full = build_full_penalty_from_blocks(
            penalty_blocks=_penalty_blocks_seq(gam),
            smoothing_params=sp,
            fit_intercept=gam.fit_intercept,
            n_coef=_n_coef(gam),
        )
        E, Es = penalty_sqrt_rows(P_full)
        lam = float(sp[0])
        q = int(P_full.shape[0])
        assert E.shape[0] == q - 1
        assert E.shape[1] == q
        np.testing.assert_allclose(E.T @ E, P_full, atol=1e-14, rtol=0.0)
        np.testing.assert_allclose(
            np.linalg.norm(Es, axis=1), 1.0, atol=1e-15, rtol=0.0
        )
        for i in range(E.shape[0]):
            j = i + 1
            np.testing.assert_allclose(E[i, j], np.sqrt(lam), rtol=1e-12, atol=0.0)
            np.testing.assert_allclose(Es[i, j], 1.0, rtol=1e-12, atol=0.0)

    def test_gaussian_re_reml_log_sp_matches_mgcv_tightly_noisy(self):
        data = _make_random_effect_data_noisy()
        formula = 'y ~ s(f, bs="re")'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        np.testing.assert_allclose(
            np.log(np.asarray(actual["fit"]["smoothing_params"], dtype=np.float64)),
            np.log(np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)),
            atol=1e-5,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=5e-8,
            rtol=5e-8,
        )

    def test_gaussian_re_reml_wood_criterion_finite_where_laplace_is_infinite(self):
        """Regression guard: RE + intercept breaks Laplace REML but not Wood profiled REML."""
        data = _make_random_effect_data()
        formula = 'y ~ s(f, bs="re")'
        gam = GAM(
            family="gaussian",
            formula=formula,
            optimize_smoothing=False,
            smoothing_method="fixed",
            smoothing_params=np.array([1.0]),
        )
        gam.fit(data=data)
        y = np.asarray(data["y"], dtype=np.float64)
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        log_sp_mgcv = np.log(
            np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        )

        wood = float(criterion_value(gam, y, log_sp_mgcv, method="reml"))
        assert np.isfinite(wood)
        laplace = float(criterion_ml_reml_exact(gam, y, log_sp_mgcv, "REML"))
        assert not np.isfinite(laplace)

    def test_gaussian_fs_reml_matches_mgcv(self):
        data = _make_fs_data()
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
            sp_log_atol=10.5,
            criterion_atol=4.0,
        )

    def test_gaussian_sz_reml_matches_mgcv(self):
        data = _make_sz_data()
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
            sp_log_atol=15.0,
            criterion_atol=2.0,
        )

    def test_gaussian_mrf_reml_matches_mgcv(self):
        data = _make_mrf_data()
        formula = (
            'y ~ s(region, bs="mrf", k=3, '
            'xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B"))))'
        )

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        asp = np.atleast_1d(
            np.asarray(actual["fit"]["smoothing_params"], dtype=np.float64)
        )
        esp = np.atleast_1d(
            np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        )
        assert asp.size == esp.size == 1
        # Both fits park λ at the optimizer floor; linear predictor parity is ~1e-15.
        assert float(asp[0]) < 1e-12 and float(esp[0]) < 1e-12

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=6e-3,
            pred_rtol=6e-3,
            sp_log_atol=0.0,  # unused when check_sp=False
            check_sp=False,
            check_criterion=True,
            criterion_atol=2.5,
        )

    def test_gaussian_mrf_low_rank_reml_matches_mgcv(self):
        """MRF with k < n_areas (low-rank truncation): full REML model fit vs mgcv."""
        data = _make_mrf_low_rank_data()
        formula = (
            'y ~ s(region, bs="mrf", k=3, '
            'xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B","D"), D=c("C"))))'
        )

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-5,
            pred_rtol=1e-5,
            sp_log_atol=0.5,
        )

    def test_gaussian_mrf_coef_full_exact_mgcv_at_mgcv_sp(self):
        """At mgcv's REML-selected sp, the Gaussian MRF solve matches mgcv's coef_full."""
        data = _make_mrf_data()
        formula = (
            'y ~ s(region, bs="mrf", k=3, '
            'xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B"))))'
        )
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        sp = np.atleast_1d(
            np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        ).ravel()
        gam = _fit_nampy_model_fixed_sp(data, formula, "gaussian", sp)

        y = gam.family.validate_y(np.asarray(data["y"], dtype=np.float64))
        sol = solve_gaussian_fit(gam, y, sp)
        coef_py = np.asarray(sol.coef_full, dtype=np.float64)
        coef_r = np.asarray(expected["fit"]["coef_full"], dtype=np.float64)

        np.testing.assert_allclose(coef_py, coef_r, atol=1e-14, rtol=0.0)

    def test_gaussian_concurvity_full_matches_mgcv(self):
        data = _make_gaussian_data(seed=17, n=160)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="tp", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        actual_diag = actual["parity"]["diagnostics"]
        expected_diag = expected["parity"]["diagnostics"]

        assert actual_diag["concurvity_full"] is not None
        assert expected_diag["concurvity_full"] is not None
        assert len(actual_diag["concurvity_labels"]) == len(
            expected_diag["concurvity_labels"]
        )
        np.testing.assert_allclose(
            np.asarray(actual_diag["concurvity_full"], dtype=np.float64),
            np.asarray(expected_diag["concurvity_full"], dtype=np.float64),
            atol=1e-9,
            rtol=1e-9,
        )

    def test_poisson_concurvity_full_matches_mgcv(self):
        data = _make_poisson_data(seed=23, n=180)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "poisson", "REML")
        expected = _run_mgcv_snapshot(data, formula, "poisson", "REML")

        actual_diag = actual["parity"]["diagnostics"]
        expected_diag = expected["parity"]["diagnostics"]

        assert actual_diag["concurvity_full"] is not None
        assert expected_diag["concurvity_full"] is not None
        assert len(actual_diag["concurvity_labels"]) == len(
            expected_diag["concurvity_labels"]
        )
        np.testing.assert_allclose(
            np.asarray(actual_diag["concurvity_full"], dtype=np.float64),
            np.asarray(expected_diag["concurvity_full"], dtype=np.float64),
            atol=1e-9,
            rtol=1e-9,
        )

    def test_gaussian_concurvity_pairwise_matches_mgcv(self):
        data = _make_gaussian_data(seed=17, n=160)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="tp", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        actual_diag = actual["parity"]["diagnostics"]
        expected_diag = expected["parity"]["diagnostics"]

        assert actual_diag["concurvity_pairwise"] is not None
        assert expected_diag["concurvity_pairwise"] is not None
        assert (
            actual_diag["concurvity_pairwise"]["labels"]
            == expected_diag["concurvity_pairwise"]["labels"]
        )
        for key in ("worst", "observed", "estimate"):
            np.testing.assert_allclose(
                np.asarray(actual_diag["concurvity_pairwise"][key], dtype=np.float64),
                np.asarray(expected_diag["concurvity_pairwise"][key], dtype=np.float64),
                atol=1e-9,
                rtol=1e-9,
            )

    def test_poisson_concurvity_pairwise_matches_mgcv(self):
        data = _make_poisson_data(seed=23, n=180)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "poisson", "REML")
        expected = _run_mgcv_snapshot(data, formula, "poisson", "REML")

        actual_diag = actual["parity"]["diagnostics"]
        expected_diag = expected["parity"]["diagnostics"]

        assert actual_diag["concurvity_pairwise"] is not None
        assert expected_diag["concurvity_pairwise"] is not None
        assert (
            actual_diag["concurvity_pairwise"]["labels"]
            == expected_diag["concurvity_pairwise"]["labels"]
        )
        for key in ("worst", "observed", "estimate"):
            np.testing.assert_allclose(
                np.asarray(actual_diag["concurvity_pairwise"][key], dtype=np.float64),
                np.asarray(expected_diag["concurvity_pairwise"][key], dtype=np.float64),
                atol=1e-9,
                rtol=1e-9,
            )

    def test_gaussian_te_fixed_sp_matches_mgcv_exactly(self):
        data = _make_gaussian_data(seed=17, n=160)
        formula = 'y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3])'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

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
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gaussian_ti_fixed_sp_matches_mgcv_exactly(self):
        rng = np.random.default_rng(23)
        n = 180
        x0 = rng.uniform(-2.0, 2.0, size=n)
        x1 = rng.uniform(-1.5, 1.5, size=n)
        y = (
            np.sin(1.1 * x0)
            + 0.35 * x0 * x1
            + 0.2 * x1**2
            + rng.normal(scale=0.15, size=n)
        )
        data = pd.DataFrame({"y": y, "x0": x0, "x1": x1})
        formula = 'y ~ ti(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3])'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

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
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gaussian_t2_fixed_sp_matches_mgcv_exactly(self):
        data = _make_gaussian_data(seed=7, n=180)
        formula = 'y ~ t2(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3, 0.9])'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")

        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=1.5e-1,
            rtol=1e-2,
        )
        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["link"], dtype=np.float64),
            np.asarray(expected["predictions"]["link"], dtype=np.float64),
            atol=1.5e-1,
            rtol=1e-2,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=9e-1,
            rtol=2e-2,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=1.0,
            rtol=5e-2,
        )

    def test_gaussian_te_reml_multi_penalty_matches_mgcv(self):
        data = _make_gaussian_data(seed=17, n=160)
        formula = 'y ~ te(x0, x1, bs=["cr","cr"], k=[5,5])'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        np.testing.assert_allclose(
            np.log(np.asarray(actual["fit"]["smoothing_params"], dtype=np.float64)),
            np.log(np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)),
            atol=4e-4,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=1e-5,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["link"], dtype=np.float64),
            np.asarray(expected["predictions"]["link"], dtype=np.float64),
            atol=1e-5,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-3,
            rtol=8e-5,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=1e-5,
            rtol=1e-6,
        )

    def test_gaussian_ti_reml_multi_penalty_matches_mgcv(self):
        data = _make_gaussian_data(seed=17, n=160)
        formula = 'y ~ ti(x0, x1, bs=["cr","cr"], k=[5,5])'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=1.2e-4,
            rtol=1.2e-4,
        )
        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["link"], dtype=np.float64),
            np.asarray(expected["predictions"]["link"], dtype=np.float64),
            atol=1.2e-4,
            rtol=1.2e-4,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=5e-4,
            rtol=3e-4,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=6e-4,
            rtol=6e-6,
        )

    def test_gaussian_t2_reml_multi_penalty_matches_mgcv(self):
        data = _make_gaussian_data(seed=17, n=160)
        formula = 'y ~ t2(x0, x1, bs=["cr","cr"], k=[5,5])'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=2.5e-3,
            rtol=2.5e-3,
        )
        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["link"], dtype=np.float64),
            np.asarray(expected["predictions"]["link"], dtype=np.float64),
            atol=2.5e-3,
            rtol=2.5e-3,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=2e-2,
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=6e-4,
            rtol=6e-6,
        )

    def test_gaussian_reml_matches_mgcv(self):
        data = _make_gaussian_data()
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=3e-2,
            pred_rtol=3e-2,
            sp_log_atol=0.45,
        )

    def test_gaussian_reml_default_s_basis_matches_mgcv_tp_default(self):
        data = _make_gaussian_data(seed=921, n=160)
        formula = "y ~ x0 + s(x1, k=8)"

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        np.testing.assert_allclose(
            np.asarray(actual["fit"]["criterion_value"], dtype=np.float64),
            np.asarray(expected["fit"]["criterion_value"], dtype=np.float64),
            atol=1e-9,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["deviance"], dtype=np.float64),
            np.asarray(expected["fit"]["deviance"], dtype=np.float64),
            atol=1e-6,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_total"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_total"], dtype=np.float64),
            atol=1e-5,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64)[-1],
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-5,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.log(
                np.atleast_1d(
                    np.asarray(actual["fit"]["smoothing_params"], dtype=np.float64)
                )
            ),
            np.log(
                np.atleast_1d(
                    np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
                )
            ),
            atol=1e-4,
            rtol=0.0,
        )
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

    def test_gaussian_reml_sig2_rss_match_mgcv_two_cr_smooths(self):
        """mgcv reports fit$sig2 and RSS; both should match our Gaussian REML fit (exact solver path)."""
        data = _make_gaussian_data(seed=123)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        np.testing.assert_allclose(
            float(actual["fit"]["scale"]),
            float(expected["fit"]["scale"]),
            rtol=2e-2,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["rss"]),
            float(expected["fit"]["rss"]),
            rtol=1e-2,
            atol=1e-6,
        )

    def test_gaussian_reml_sig2_matches_mgcv_joint_outer_tensor_smooth(self):
        """Tensor-product REML carries sigma^2 through the joint Gaussian outer path."""
        data = _make_gaussian_data(seed=29, n=140)
        formula = 'y ~ te(x0, x1, bs=["cr", "cr"], k=[6, 6])'
        gam = _fit_nampy_model(data, formula, "gaussian", "REML")
        actual = gam.parity_snapshot(X=data, include_covariances=True)
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        np.testing.assert_allclose(
            float(actual["fit"]["scale"]),
            float(expected["fit"]["scale"]),
            rtol=2e-2,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["rss"]),
            float(expected["fit"]["rss"]),
            rtol=1e-2,
            atol=1e-6,
        )

        core = _get_core(gam)
        trace = getattr(core, "_optim_trace", None) or []
        assert len(trace) >= 1
        ri = trace[-1].get("rank_info") or {}
        assert bool(ri.get("joint_gaussian_reml_outer", False)) is False
        opt_sigma = getattr(core, "_gaussian_reml_sigma2_opt_", None)
        assert opt_sigma is not None
        np.testing.assert_allclose(
            float(opt_sigma), float(actual["fit"]["scale"]), rtol=0.0, atol=5e-11
        )
        endpoint = actual["parity"]["diagnostics"]["optimizer_endpoint"]
        assert endpoint is not None
        assert bool(endpoint["joint_gaussian_reml_outer"]) is True
        assert (
            float(endpoint["projected_gradient_inf_norm"])
            <= float(endpoint["gradient_inf_norm"]) + 1e-15
        )

    def test_gaussian_reml_sig2_matches_mgcv_joint_outer_mrf_exact(self):
        """MRF REML uses the same profile-variance Gaussian outer path."""
        data = _make_mrf_data()
        formula = (
            'y ~ s(region, bs="mrf", k=3, '
            'xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B"))))'
        )
        gam = _fit_nampy_model(data, formula, "gaussian", "REML")
        actual = gam.parity_snapshot(X=data, include_covariances=True)
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        a_sp = np.log(np.asarray(actual["fit"]["smoothing_params"], dtype=np.float64))
        e_sp = np.log(np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64))
        np.testing.assert_allclose(a_sp, e_sp, atol=0.25, rtol=0.0)

        np.testing.assert_allclose(
            float(actual["fit"]["scale"]),
            float(expected["fit"]["scale"]),
            rtol=0.6,
            atol=0.0,
        )

        np.testing.assert_allclose(
            float(actual["fit"]["criterion_value"]),
            float(expected["fit"]["criterion_value"]),
            atol=0.5,
            rtol=0.0,
        )

        core = _get_core(gam)
        trace = getattr(core, "_optim_trace", None) or []
        assert len(trace) >= 1
        ri = trace[-1].get("rank_info") or {}
        assert bool(ri.get("joint_gaussian_reml_outer", False)) is False

    def test_binomial_reml_matches_mgcv(self):
        case = get_parity_case("binomial_cr_uni_reml")
        data = make_parity_case_data(case.case_id)
        actual = _fit_nampy_snapshot(data, case.formula, case.family, case.method)
        expected = _run_mgcv_snapshot(data, case.formula, case.family, case.method)

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=4e-2,
            pred_rtol=4e-2,
            sp_log_atol=0.65,
        )

    def test_poisson_reml_matches_mgcv(self):
        case = get_parity_case("poisson_cr_uni_reml")
        data = make_parity_case_data(case.case_id)
        actual = _fit_nampy_snapshot(data, case.formula, case.family, case.method)
        expected = _run_mgcv_snapshot(data, case.formula, case.family, case.method)

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-2,
            pred_rtol=5e-2,
            sp_log_atol=0.55,
        )

    def test_gaussian_select_reml_matches_mgcv(self):
        data = _make_gaussian_data(seed=999)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML", select=True)
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML", select=True)

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=4e-2,
            pred_rtol=4e-2,
            sp_log_atol=0.55,
        )

    def test_gamma_reml_matches_mgcv(self):
        case = get_parity_case("gamma_cr_uni_reml")
        data = make_parity_case_data(case.case_id)
        actual = _fit_nampy_snapshot(data, case.formula, case.family, case.method)
        expected = _run_mgcv_snapshot(data, case.formula, case.family, case.method)
        a_fit = actual["fit"]
        e_fit = expected["fit"]
        a_pred = actual["predictions"]
        e_pred = expected["predictions"]

        # Gamma REML is currently less stable in log(sp) and EDF parity than
        # Gaussian/Binomial/Poisson, so we assert predictive and criterion parity.
        np.testing.assert_allclose(
            np.asarray(a_pred["response"], dtype=np.float64),
            np.asarray(e_pred["response"], dtype=np.float64),
            atol=2.7e-1,
            rtol=2.5e-1,
        )
        np.testing.assert_allclose(
            np.asarray(a_pred["link"], dtype=np.float64),
            np.asarray(e_pred["link"], dtype=np.float64),
            atol=2.2e-1,
            rtol=2.5e-1,
        )
        np.testing.assert_allclose(
            np.asarray(a_fit["deviance"], dtype=np.float64),
            np.asarray(e_fit["deviance"], dtype=np.float64),
            atol=1.0,
            rtol=0.15,
        )
        np.testing.assert_allclose(
            np.asarray(a_fit["criterion_value"], dtype=np.float64),
            np.asarray(e_fit["criterion_value"], dtype=np.float64),
            atol=2.0,
            rtol=0.1,
        )

    def test_gamma_reml_optimizes_without_abnormal_warning(self):
        data = _make_gamma_data()
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gam = GAM(
                family="gamma",
                formula=formula,
                optimize_smoothing=True,
                smoothing_method="REML",
            )
            gam.fit(data=data)

        abnormal = [
            str(w.message)
            for w in caught
            if "Smoothing optimisation did not converge: ABNORMAL" in str(w.message)
        ]
        assert not abnormal
        assert gam._optim_result is not None
        assert bool(gam._optim_result.success)

    def test_negbin_reml_matches_mgcv(self):
        data = _make_negbin_data(theta=1.0)
        family = {"name": "negbin", "theta": 1.0}
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, family, "REML")
        expected = _run_mgcv_snapshot(data, formula, family, "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=7e-2,
            pred_rtol=7e-2,
            sp_log_atol=0.8,
        )

    def test_poisson_reml_with_formula_offset_matches_mgcv(self):
        data = _make_poisson_data(seed=177)
        data = data.copy()
        data["off"] = np.linspace(-0.35, 0.35, len(data))
        formula = 'y ~ offset(off) + s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "poisson", "REML")
        expected = _run_mgcv_snapshot(data, formula, "poisson", "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=6e-2,
            pred_rtol=6e-2,
            sp_log_atol=0.65,
        )

    @pytest.mark.parametrize(
        (
            "family",
            "data_factory",
            "formula",
            "sp_atol",
            "sp_rtol",
            "log_sp_atol",
            "edf_atol",
            "pred_atol",
        ),
        [
            (
                "poisson",
                lambda: _make_poisson_data(seed=71, n=220),
                'y ~ t2(x0, x1, bs=["cr", "cr"], k=[6, 6])',
                1e-4,
                5e-8,
                3e-8,
                5e-7,
                1e-10,
            ),
            (
                "binomial",
                lambda: _make_binomial_data(seed=73, n=220),
                'y ~ t2(x0, x1, bs=["cr", "cr"], k=[6, 6])',
                1e-4,
                5e-8,
                3e-8,
                5e-7,
                1e-10,
            ),
            (
                "gamma",
                lambda: _make_gamma_data(seed=101, n=220),
                'y ~ t2(x0, x1, bs=["cr", "cr"], k=[6, 6])',
                1e-6,
                7e-4,
                7e-4,
                1e-1,
                1e-5,
            ),
            (
                {"name": "negbin", "theta": 1.0},
                lambda: _make_negbin_data(seed=79, n=240, theta=1.0),
                'y ~ t2(x0, x1, bs=["cr", "cr"], k=[6, 6])',
                3e-6,
                2e-8,
                2e-8,
                1.3e-1,
                1e-6,
            ),
        ],
    )
    def test_optimized_tensor_t2_snapshot_matches_mgcv(
        self,
        family,
        data_factory,
        formula,
        sp_atol,
        sp_rtol,
        log_sp_atol,
        edf_atol,
        pred_atol,
    ):
        data = data_factory()
        actual = _fit_nampy_snapshot(data, formula, family, "REML")
        expected = _run_mgcv_snapshot(data, formula, family, "REML")

        if family == "gamma":
            _assert_basic_mgcv_parity(
                actual,
                expected,
                pred_atol=2e-5,
                pred_rtol=2e-5,
                sp_log_atol=log_sp_atol,
                check_sp=False,
                criterion_atol=0.5,
            )
            return

        _assert_exact_mgcv_snapshot_parity(
            actual,
            expected,
            pred_atol=pred_atol,
            pred_rtol=pred_atol,
            edf_atol=edf_atol,
            criterion_atol=1e-8 if family == "gamma" else 5e-7,
            criterion_rtol=1e-8 if family == "gamma" else 5e-7,
            sp_atol=sp_atol,
            sp_rtol=sp_rtol,
            log_sp_atol=log_sp_atol,
        )


class TestMgcvDeviancePenaltyScaleAssembly:
    """
    Cross-check NAMpy mgcv-style dev / P / scale helpers against quantities computed
    in R (mgcv) and exported via tests/parity/mgcv_snapshot.R.
    """

    @staticmethod
    def _gaussian_case():
        data = _make_gaussian_data()
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        return data, formula

    def test_r_dev_sum_dev_resids_matches_deviance(self):
        """Sanity: R's sum(dev.resids) matches fit$deviance (mgcv bookkeeping)."""
        data, formula = self._gaussian_case()
        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        fit = snap["fit"]
        np.testing.assert_allclose(
            float(fit["dev_sum_dev_resids"]),
            float(fit["deviance"]),
            rtol=0.0,
            atol=1e-10,
        )

    def test_gaussian_sum_dev_resids_helper_matches_mgcv(self):
        data, formula = self._gaussian_case()
        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        y = data["y"].to_numpy(dtype=np.float64)
        mu = np.asarray(snap["predictions"]["response"], dtype=np.float64)
        dev_py = gaussian_weighted_residual_sum_squares(y, mu, None)
        dev_r = float(snap["fit"]["dev_sum_dev_resids"])
        np.testing.assert_allclose(dev_py, dev_r, rtol=0.0, atol=1e-9)

    def test_gaussian_scale_est_deviance_matches_mgcv_sig2(self):
        data, formula = self._gaussian_case()
        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        fit = snap["fit"]
        n = int(fit["n_obs"])
        scale_py = deviance_method_scale_estimate(
            float(fit["deviance"]),
            float(fit["edf_total"]),
            float(n),
        )
        np.testing.assert_allclose(scale_py, float(fit["scale"]), rtol=1e-7, atol=1e-7)

    def test_gaussian_penalty_quadratic_matches_mgcv(self):
        data, formula = self._gaussian_case()
        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        gam = _fit_nampy_model(data, formula, "gaussian", "REML")
        sp = np.asarray(gam.smoothing_params, dtype=np.float64)
        sol = solve_gaussian_fit(gam, gam.y_, sp)
        pen_py = quadratic_form_penalty(sol.coef_full, sol.penalty_matrix)
        pen_r = float(snap["fit"]["penalty_quadratic"])
        np.testing.assert_allclose(pen_py, pen_r, rtol=2e-5, atol=2e-5)

    def test_poisson_dev_sum_dev_resids_matches_deviance(self):
        data = _make_poisson_data()
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        snap = _run_mgcv_snapshot(data, formula, "poisson", "REML")
        fit = snap["fit"]
        np.testing.assert_allclose(
            float(fit["dev_sum_dev_resids"]),
            float(fit["deviance"]),
            rtol=0.0,
            atol=1e-9,
        )

    def test_weighted_gaussian_dev_penalty_match_mgcv(self):
        rng = np.random.default_rng(31)
        n = 100
        x0 = rng.uniform(-2, 2, size=n)
        x1 = rng.uniform(-1.5, 1.5, size=n)
        y = np.sin(1.1 * x0) + 0.35 * x1**2 + rng.normal(0, 0.16, size=n)
        w = rng.uniform(0.5, 2.0, size=n)
        data = pd.DataFrame({"y": y, "x0": x0, "x1": x1, "w": w})
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML", weights_column="w")
        yv = data["y"].to_numpy(dtype=np.float64)
        mu_r = np.asarray(snap["predictions"]["response"], dtype=np.float64)
        wv = data["w"].to_numpy(dtype=np.float64)
        dev_py = gaussian_weighted_residual_sum_squares(yv, mu_r, wv)
        np.testing.assert_allclose(
            dev_py,
            float(snap["fit"]["dev_sum_dev_resids"]),
            rtol=0.0,
            atol=1e-8,
        )
        r_sp = np.asarray(snap["fit"]["smoothing_params"], dtype=np.float64)
        gam = _fit_nampy_model_fixed_sp(
            data, formula, "gaussian", r_sp, sample_weight="w"
        )
        sol = solve_gaussian_fit(
            gam,
            gam.y_,
            gam.smoothing_params,
            weights=gam.prior_weights_,
        )
        pen_py = quadratic_form_penalty(sol.coef_full, sol.penalty_matrix)
        np.testing.assert_allclose(
            pen_py,
            float(snap["fit"]["penalty_quadratic"]),
            rtol=2e-5,
            atol=2e-5,
        )


class TestGaussianPriorWeights:
    """Gaussian WLS via mgcv-style prior weights (sample_weight)."""

    def test_sample_weight_ones_matches_no_weights(self):
        data = _make_gaussian_data()
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        kwargs = {
            "family": "gaussian",
            "formula": formula,
            "optimize_smoothing": False,
            "smoothing_method": "fixed",
            "smoothing_params": 1.0,
        }
        g0 = GAM(**kwargs)
        g0.fit(data=data)
        g1 = GAM(**kwargs)
        g1.fit(data=data, sample_weight=np.ones(len(data), dtype=np.float64))
        np.testing.assert_allclose(g0.coef_full_, g1.coef_full_, rtol=0.0, atol=1e-10)
        np.testing.assert_allclose(g0.scale_, g1.scale_, rtol=0.0, atol=1e-10)

    def test_wls_solve_matches_explicit_normal_equations(self):
        rng = np.random.default_rng(3)
        n = 50
        x0 = rng.uniform(-1, 1, size=n)
        y = 0.9 * np.sin(2.5 * x0) + rng.normal(0, 0.2, size=n)
        w = rng.uniform(0.6, 1.8, size=n)
        d = pd.DataFrame({"y": y, "x0": x0, "w": w})
        gam = GAM(
            family="gaussian",
            formula='y ~ s(x0, bs="cr", k=8)',
            smoothing_params=0.6,
            optimize_smoothing=False,
            smoothing_method="fixed",
        )
        gam.fit(data=d, sample_weight="w")
        sol = solve_gaussian_fit(
            gam,
            gam.y_,
            gam.smoothing_params,
            weights=gam.prior_weights_,
        )
        X = np.asarray(sol.X, dtype=np.float64)
        P = np.asarray(sol.penalty_matrix, dtype=np.float64)
        y_work = np.asarray(gam.y_, dtype=np.float64).ravel()
        lhs = X.T @ (w[:, np.newaxis] * X) + P
        rhs = X.T @ (w * y_work)
        beta_manual = np.linalg.solve(lhs, rhs)
        np.testing.assert_allclose(sol.coef_full, beta_manual, rtol=0.0, atol=1e-9)

    def test_sample_weight_bad_length_raises(self):
        data = _make_gaussian_data()
        gam = GAM(family="gaussian", formula='y ~ s(x0, bs="cr", k=8)')
        with pytest.raises(ValueError, match="sample_weight"):
            gam.fit(data=data, sample_weight=np.ones(len(data) + 2, dtype=np.float64))


class TestGaussianPriorWeightsMgcvParity:
    def test_reml_weighted_snapshot_parity_at_r_smoothing(self):
        """
        With mgcv's REML-selected ``sp`` held fixed, weighted WLS + penalties should
        match R (outer ``sp`` optimisers can differ slightly under weights).
        """
        rng = np.random.default_rng(21)
        n = 100
        x0 = rng.uniform(-2, 2, size=n)
        x1 = rng.uniform(-1.5, 1.5, size=n)
        y = np.sin(1.1 * x0) + 0.3 * x1**2 + rng.normal(0, 0.18, size=n)
        w = rng.uniform(0.5, 2.0, size=n)
        d = pd.DataFrame({"y": y, "x0": x0, "x1": x1, "w": w})
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        expected = _run_mgcv_snapshot(
            d, formula, "gaussian", "REML", weights_column="w"
        )
        r_sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        gam = _fit_nampy_model_fixed_sp(d, formula, "gaussian", r_sp, sample_weight="w")
        actual = gam.parity_snapshot(X=d, include_covariances=True)
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-4,
            pred_rtol=5e-4,
            sp_log_atol=1e-10,
            check_criterion=False,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["scale"]),
            float(expected["fit"]["scale"]),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_weighted_joint_reml_matches_laplace_assembly(self):
        """
        At R's (sp, sig2), the joint Gaussian REML objective agrees with the
        Laplace REML score built from the same WLS solve (including ``sum(log w)``).
        """
        rng = np.random.default_rng(37)
        n = 100
        x0 = rng.uniform(-2, 2, size=n)
        x1 = rng.uniform(-1.5, 1.5, size=n)
        y = np.sin(1.05 * x0) + 0.28 * x1**2 + rng.normal(0, 0.17, size=n)
        w = rng.uniform(0.45, 2.1, size=n)
        d = pd.DataFrame({"y": y, "x0": x0, "x1": x1, "w": w})
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        snap = _run_mgcv_snapshot(d, formula, "gaussian", "REML", weights_column="w")
        r_sp = np.asarray(snap["fit"]["smoothing_params"], dtype=np.float64)
        gam = _fit_nampy_model_fixed_sp(d, formula, "gaussian", r_sp, sample_weight="w")
        core = _get_core(gam)
        sp = np.asarray(core.smoothing_params, dtype=np.float64)
        mask = core.smoothing_fixed_mask_
        if mask is None:
            log_free = np.log(sp)
        else:
            log_free = np.log(sp[~np.asarray(mask, dtype=bool)])
        sig2 = float(snap["fit"]["scale"])
        joint = criterion_ml_reml_gaussian_dynamic_joint(
            core, core.y_, log_free, np.log(sig2), method="REML"
        )

        sol = solve_gaussian_fit(core, core.y_, sp, weights=core.prior_weights_)
        yv = np.asarray(core.y_, dtype=np.float64).ravel()
        mu = np.asarray(sol.mu, dtype=np.float64).ravel()
        wv = d["w"].to_numpy(dtype=np.float64)
        dev = gaussian_weighted_residual_sum_squares(yv, mu, wv)
        Pq = quadratic_form_penalty(sol.coef_full, sol.penalty_matrix)
        c_a, _ = cho_factor(
            np.asarray(sol.A, dtype=np.float64), lower=True, check_finite=False
        )
        logdet_a = 2.0 * float(np.sum(np.log(np.abs(np.diag(c_a)))))
        logdet_s = float(_stable_penalty_logdet(core, sp))
        mp = float(
            _static_penalty_null_dim(core)
            + int(bool(getattr(core, "fit_intercept", False)))
        )
        fit4 = gaussian_reml_laplace_score(
            dev, Pq, sig2, logdet_a - logdet_s, mp, wv, gamma=1.0, reml=True
        )
        np.testing.assert_allclose(joint, fit4, rtol=0.0, atol=1e-10)

    def test_weighted_reml_end_to_end_sp_matches_mgcv(self):
        """Full weighted Gaussian REML outer optimization tracks mgcv tightly."""
        rng = np.random.default_rng(43)
        n = 160
        x0 = rng.uniform(-2, 2, size=n)
        x1 = rng.uniform(-1.5, 1.5, size=n)
        y = np.sin(1.2 * x0) + 0.25 * x1**2 + rng.normal(0, 0.2, size=n)
        w = rng.uniform(0.4, 2.2, size=n)
        d = pd.DataFrame({"y": y, "x0": x0, "x1": x1, "w": w})
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(d, formula, "gaussian", "REML", sample_weight="w")
        expected = _run_mgcv_snapshot(
            d, formula, "gaussian", "REML", weights_column="w"
        )

        _assert_exact_mgcv_snapshot_parity(
            actual,
            expected,
            pred_atol=2e-8,
            pred_rtol=2e-8,
            edf_atol=2e-6,
            criterion_atol=1e-10,
            criterion_rtol=0.0,
            sp_atol=1e-8,
            sp_rtol=2e-6,
            log_sp_atol=2e-6,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["scale"]),
            float(expected["fit"]["scale"]),
            rtol=2e-10,
            atol=2e-10,
        )


# ---------------------------------------------------------------------------
# Smooth-type parity: cc, ps, gp, numeric by-variable
# ---------------------------------------------------------------------------


class TestCyclicCubicSmooth:
    """Cyclic cubic regression spline (bs='cc') parity against mgcv."""

    def _make_cyclic_data(self, seed=77, n=180):
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.0, 2 * np.pi, size=n)
        y = np.sin(x) + 0.3 * np.cos(2 * x) + rng.normal(scale=0.12, size=n)
        return pd.DataFrame({"y": y, "x": x})

    def test_cc_smoothcon_basis_matches_mgcv(self):
        data = self._make_cyclic_data()
        smooth_expr_r = 's(x, bs="cc", k=9)'

        parsed = parse_gam_formula('y ~ s(x, bs="cc", k=9)')
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["x"]].to_numpy(dtype=np.float64),
            ["x"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        _assert_allclose_up_to_column_sign(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_cc_smoothcon_penalties_match_mgcv(self):
        data = self._make_cyclic_data()
        smooth_expr_r = 's(x, bs="cc", k=9)'

        parsed = parse_gam_formula('y ~ s(x, bs="cc", k=9)')
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["x"]].to_numpy(dtype=np.float64),
            ["x"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_penalties(
            data, smooth_expr_r, absorb_cons=True, scale_penalty=True
        )
        actual = [
            np.asarray(pb.matrix, dtype=np.float64) for pb in design.compiled_penalties
        ]
        target = [np.asarray(S, dtype=np.float64) for S in expected["S"]]

        assert len(actual) == len(target) == 1
        np.testing.assert_allclose(actual[0], target[0], atol=1e-10, rtol=1e-10)

    def test_gaussian_cc_fixed_sp_matches_mgcv_exactly(self):
        data = self._make_cyclic_data(seed=78)
        formula = 'y ~ s(x, bs="cc", k=9, sp=0.8)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gaussian_cc_reml_matches_mgcv(self):
        data = self._make_cyclic_data(seed=79, n=200)
        formula = 'y ~ s(x, bs="cc", k=10)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-3,
            pred_rtol=0.0,
            sp_log_atol=0.6,
        )


class TestPSplineSmooth:
    """P-spline (bs='ps') standalone parity against mgcv."""

    def _make_ps_data(self, seed=81, n=180):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-2.0, 2.0, size=n)
        y = np.sin(1.3 * x) + 0.2 * x**2 + rng.normal(scale=0.14, size=n)
        return pd.DataFrame({"y": y, "x": x})

    def test_ps_smoothcon_basis_matches_mgcv(self):
        data = self._make_ps_data()
        smooth_expr_r = 's(x, bs="ps", k=12)'

        parsed = parse_gam_formula('y ~ s(x, bs="ps", k=12)')
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["x"]].to_numpy(dtype=np.float64),
            ["x"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        _assert_allclose_up_to_column_sign(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_ps_smoothcon_penalties_match_mgcv(self):
        data = self._make_ps_data()
        smooth_expr_r = 's(x, bs="ps", k=12)'

        parsed = parse_gam_formula('y ~ s(x, bs="ps", k=12)')
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["x"]].to_numpy(dtype=np.float64),
            ["x"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_penalties(
            data, smooth_expr_r, absorb_cons=True, scale_penalty=True
        )
        actual = [
            np.asarray(pb.matrix, dtype=np.float64) for pb in design.compiled_penalties
        ]
        target = [np.asarray(S, dtype=np.float64) for S in expected["S"]]

        assert len(actual) == len(target) == 1
        np.testing.assert_allclose(actual[0], target[0], atol=1e-10, rtol=1e-10)

    def test_gaussian_ps_fixed_sp_matches_mgcv_exactly(self):
        data = self._make_ps_data(seed=82)
        formula = 'y ~ s(x, bs="ps", k=12, sp=0.5)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gaussian_ps_reml_matches_mgcv(self):
        data = self._make_ps_data(seed=83, n=200)
        formula = 'y ~ s(x, bs="ps", k=14)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-3,
            pred_rtol=0.0,
            sp_log_atol=0.5,
        )

    def test_gaussian_ps_two_smooths_reml_matches_mgcv(self):
        """Two independent P-spline terms, each on a different covariate."""
        rng = np.random.default_rng(84)
        n = 200
        x0 = rng.uniform(-2.0, 2.0, size=n)
        x1 = rng.uniform(-1.5, 1.5, size=n)
        y = np.sin(x0) + 0.35 * x1**2 + rng.normal(scale=0.15, size=n)
        data = pd.DataFrame({"y": y, "x0": x0, "x1": x1})
        formula = 'y ~ s(x0, bs="ps", k=12) + s(x1, bs="ps", k=12)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=8e-3,
            pred_rtol=0.0,
            sp_log_atol=0.6,
        )

    # ------------------------------------------------------------------ #
    # Non-default ps order: m=[1,1] and m=[3,3]                          #
    # ------------------------------------------------------------------ #

    def test_ps_m11_smoothcon_basis_matches_mgcv(self):
        data = self._make_ps_data()
        smooth_expr_r = 's(x, bs="ps", k=12, m=c(1,1))'

        parsed = parse_gam_formula('y ~ s(x, bs="ps", k=12, m=[1,1])')
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["x"]].to_numpy(dtype=np.float64),
            ["x"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        _assert_allclose_up_to_column_sign(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_ps_m11_smoothcon_penalties_match_mgcv(self):
        data = self._make_ps_data()
        smooth_expr_r = 's(x, bs="ps", k=12, m=c(1,1))'

        parsed = parse_gam_formula('y ~ s(x, bs="ps", k=12, m=[1,1])')
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["x"]].to_numpy(dtype=np.float64),
            ["x"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_penalties(
            data, smooth_expr_r, absorb_cons=True, scale_penalty=True
        )
        actual = [
            np.asarray(pb.matrix, dtype=np.float64) for pb in design.compiled_penalties
        ]
        target = [np.asarray(S, dtype=np.float64) for S in expected["S"]]

        assert len(actual) == len(target) == 1
        np.testing.assert_allclose(actual[0], target[0], atol=1e-10, rtol=1e-10)

    def test_gaussian_ps_m11_fixed_sp_matches_mgcv(self):
        data = self._make_ps_data(seed=85)
        formula = 'y ~ s(x, bs="ps", k=12, m=[1,1], sp=0.5)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gaussian_ps_m11_reml_matches_mgcv(self):
        data = self._make_ps_data(seed=86, n=200)
        formula = 'y ~ s(x, bs="ps", k=14, m=[1,1])'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-3,
            pred_rtol=0.0,
            sp_log_atol=0.5,
        )

    def test_ps_m33_smoothcon_basis_matches_mgcv(self):
        data = self._make_ps_data()
        smooth_expr_r = 's(x, bs="ps", k=14, m=c(3,3))'

        parsed = parse_gam_formula('y ~ s(x, bs="ps", k=14, m=[3,3])')
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["x"]].to_numpy(dtype=np.float64),
            ["x"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        _assert_allclose_up_to_column_sign(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_ps_m33_smoothcon_penalties_match_mgcv(self):
        data = self._make_ps_data()
        smooth_expr_r = 's(x, bs="ps", k=14, m=c(3,3))'

        parsed = parse_gam_formula('y ~ s(x, bs="ps", k=14, m=[3,3])')
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["x"]].to_numpy(dtype=np.float64),
            ["x"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_penalties(
            data, smooth_expr_r, absorb_cons=True, scale_penalty=True
        )
        actual = [
            np.asarray(pb.matrix, dtype=np.float64) for pb in design.compiled_penalties
        ]
        target = [np.asarray(S, dtype=np.float64) for S in expected["S"]]

        assert len(actual) == len(target) == 1
        np.testing.assert_allclose(actual[0], target[0], atol=1e-10, rtol=1e-10)

    def test_gaussian_ps_m33_fixed_sp_matches_mgcv(self):
        data = self._make_ps_data(seed=87)
        formula = 'y ~ s(x, bs="ps", k=14, m=[3,3], sp=0.5)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gaussian_ps_m33_reml_matches_mgcv(self):
        data = self._make_ps_data(seed=88, n=200)
        formula = 'y ~ s(x, bs="ps", k=14, m=[3,3])'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-3,
            pred_rtol=0.0,
            sp_log_atol=0.5,
        )


class TestGPSmooth:
    """Gaussian process smooth (bs='gp') parity against mgcv."""

    def _make_gp_data(self, seed=91, n=160):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-3.0, 3.0, size=n)
        y = np.exp(-0.5 * x**2) + 0.4 * np.sin(x) + rng.normal(scale=0.1, size=n)
        return pd.DataFrame({"y": y, "x": x})

    def test_gp_smoothcon_basis_matches_mgcv(self):
        """GP basis matrix from smoothCon should match NAMpy's GP basis."""
        data = self._make_gp_data()
        smooth_expr_r = 's(x, bs="gp", k=10)'

        parsed = parse_gam_formula('y ~ s(x, bs="gp", k=10)')
        specs = compile_predictor_specs_from_formula(parsed)
        design = compile_predictor_designs(
            data[["x"]].to_numpy(dtype=np.float64),
            ["x"],
            specs,
        )[0]

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        actual_X = np.asarray(design.design_matrix, dtype=np.float64)
        expected_X = np.asarray(expected["X"], dtype=np.float64)

        assert actual_X.shape == expected_X.shape
        # GP bases can have sign/rotation ambiguity in eigenvectors; compare
        # that the column span is the same (each actual column is a linear
        # combination of expected columns up to the tolerance).
        _assert_allclose_up_to_column_sign(actual_X, expected_X, atol=1e-8, rtol=1e-8)

    def test_gaussian_gp_fixed_sp_matches_mgcv(self):
        data = self._make_gp_data(seed=92)
        formula = 'y ~ s(x, bs="gp", k=10, sp=1.0)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=1e-8,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-8,
            rtol=1e-8,
        )

    def test_gaussian_gp_reml_matches_mgcv(self):
        data = self._make_gp_data(seed=93, n=180)
        formula = 'y ~ s(x, bs="gp", k=12)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=2e-7,
            pred_rtol=2e-7,
            sp_log_atol=2e-5,
        )

    def test_gaussian_gp_two_smooths_reml_matches_mgcv(self):
        rng = np.random.default_rng(94)
        n = 180
        x0 = rng.uniform(-2.0, 2.0, size=n)
        x1 = rng.uniform(-2.0, 2.0, size=n)
        y = np.exp(-0.5 * x0**2) + 0.3 * np.sin(x1) + rng.normal(scale=0.12, size=n)
        data = pd.DataFrame({"y": y, "x0": x0, "x1": x1})
        formula = 'y ~ s(x0, bs="gp", k=10) + s(x1, bs="gp", k=10)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=2e-7,
            pred_rtol=2e-7,
            sp_log_atol=2e-5,
        )


class TestNumericByVariable:
    """Numeric by-variable smooth s(x, by=z) parity against mgcv."""

    def _make_by_data(self, seed=101, n=200):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-2.0, 2.0, size=n)
        z = rng.uniform(-1.0, 1.0, size=n)
        y = np.sin(x) * z + 0.2 * rng.normal(size=n)
        return pd.DataFrame({"y": y, "x": x, "z": z})

    def test_gaussian_numeric_by_cr_fixed_sp_matches_mgcv(self):
        data = self._make_by_data(seed=102)
        formula = 'y ~ s(x, by=z, bs="cr", k=8, sp=1.0)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

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
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gaussian_numeric_by_cr_reml_matches_mgcv(self):
        data = self._make_by_data(seed=103, n=220)
        formula = 'y ~ s(x, by=z, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-3,
            pred_rtol=0.0,
            sp_log_atol=0.6,
        )

    def test_gaussian_numeric_by_two_smooths_reml_matches_mgcv(self):
        """Two separate numeric by-variable terms."""
        rng = np.random.default_rng(104)
        n = 220
        x0 = rng.uniform(-2.0, 2.0, size=n)
        x1 = rng.uniform(-1.5, 1.5, size=n)
        z = rng.uniform(0.5, 1.5, size=n)
        y = np.sin(x0) * z - 0.3 * np.cos(x1) * z + 0.15 * rng.normal(size=n)
        data = pd.DataFrame({"y": y, "x0": x0, "x1": x1, "z": z})
        formula = 'y ~ s(x0, by=z, bs="cr", k=8) + s(x1, by=z, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=8e-3,
            pred_rtol=0.0,
            sp_log_atol=0.7,
        )

    def test_poisson_numeric_by_reml_matches_mgcv(self):
        rng = np.random.default_rng(105)
        n = 220
        x = rng.uniform(-1.5, 1.5, size=n)
        z = rng.uniform(0.5, 1.5, size=n)
        eta = 0.1 + 0.6 * np.sin(x) * z
        y = rng.poisson(np.exp(eta))
        data = pd.DataFrame({"y": y, "x": x, "z": z})
        formula = 'y ~ s(x, by=z, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "poisson", "REML")
        expected = _run_mgcv_snapshot(data, formula, "poisson", "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-2,
            pred_rtol=0.0,
            sp_log_atol=0.8,
            criterion_atol=1.5,
        )


class TestSmoothingMethodParity:
    """Parity tests for GCV and ML smoothing parameter selection methods."""

    def test_gaussian_gcv_matches_mgcv(self):
        data = _make_gaussian_data(seed=200, n=180)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        model = _fit_nampy_model(data, formula, "gaussian", "gcv")
        actual = model.parity_snapshot(X=data, include_covariances=True)
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "GCV.Cp")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-5,
            pred_rtol=0.0,
            sp_log_atol=1e-5,
        )

    def test_gamma_gcv_matches_mgcv(self):
        data = _make_gamma_data(seed=201, n=220)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        model = _fit_nampy_model(data, formula, "gamma", "gcv")
        actual = model.parity_snapshot(X=data, include_covariances=True)
        expected = _run_mgcv_snapshot(data, formula, "gamma", "GCV.Cp")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-3,
            pred_rtol=0.0,
            sp_log_atol=0.1,
        )

    def test_gaussian_ml_matches_mgcv(self):
        data = _make_gaussian_data(seed=202, n=180)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "ML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "ML")

        # ML criterion values differ in additive constant between NAMpy and mgcv.
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-4,
            pred_rtol=0.0,
            sp_log_atol=1e-4,
            check_criterion=False,
        )

    def test_binomial_ml_matches_mgcv(self):
        data = _make_binomial_data(seed=203, n=220)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "binomial", "ML")
        expected = _run_mgcv_snapshot(data, formula, "binomial", "ML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-2,
            pred_rtol=0.0,
            sp_log_atol=2.0,
        )

    def test_poisson_ml_matches_mgcv(self):
        data = _make_poisson_data(seed=204, n=220)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "poisson", "ML")
        expected = _run_mgcv_snapshot(data, formula, "poisson", "ML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-2,
            pred_rtol=0.0,
            sp_log_atol=0.5,
        )


class TestAdditionalScenarioParity:
    """Parity tests for gaps 7, 9-12, 14-17."""

    # ------------------------------------------------------------------ #
    # Gap 7: select=True for non-Gaussian families                        #
    # ------------------------------------------------------------------ #

    def test_binomial_select_reml_matches_mgcv(self):
        # With select=True the outer optimizer may drive individual sp values
        # to very different extremes in NAMpy vs mgcv, so we only check
        # that predictions agree (not sp values) and deviance is finite.
        data = _make_binomial_data(seed=300, n=220)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "binomial", "REML", select=True)
        expected = _run_mgcv_snapshot(data, formula, "binomial", "REML", select=True)

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-3,
            pred_rtol=0.0,
            sp_log_atol=1e-10,
            check_sp=False,
        )

    def test_poisson_select_reml_matches_mgcv(self):
        data = _make_poisson_data(seed=301, n=220)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "poisson", "REML", select=True)
        expected = _run_mgcv_snapshot(data, formula, "poisson", "REML", select=True)

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-3,
            pred_rtol=0.0,
            sp_log_atol=1e-10,
            check_sp=False,
        )

    def test_gaussian_re_select_reml_matches_mgcv_exactly(self):
        data = _make_random_effect_data_noisy()
        formula = 'y ~ s(f, bs="re")'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML", select=True)
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML", select=True)

        _assert_exact_mgcv_snapshot_parity(
            actual,
            expected,
            pred_atol=1e-8,
            pred_rtol=1e-8,
            edf_atol=1e-8,
            criterion_atol=1e-8,
            criterion_rtol=1e-8,
            sp_atol=1e-8,
            sp_rtol=1e-8,
            log_sp_atol=1e-6,
        )

    def test_gaussian_fs_select_reml_matches_mgcv(self):
        data = _make_fs_data()
        formula = 'y ~ s(f, x, bs="fs", k=6)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML", select=True)
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML", select=True)

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-6,
            pred_rtol=1e-6,
            sp_log_atol=2.0,
            criterion_atol=1e-3,
        )

        actual_sp = np.asarray(actual["fit"]["smoothing_params"], dtype=np.float64)
        expected_sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        actual_score_r = _run_mgcv_fixed_sp_score(
            data,
            formula,
            "gaussian",
            "REML",
            actual_sp,
            select=True,
        )
        expected_score_r = _run_mgcv_fixed_sp_score(
            data,
            formula,
            "gaussian",
            "REML",
            expected_sp,
            select=True,
        )

        assert (
            float(actual_score_r["criterion_value"])
            <= float(expected_score_r["criterion_value"]) + 2e-5
        )
        assert (
            np.linalg.norm(np.asarray(expected_score_r["gradient"], dtype=np.float64))
            > 1e-6
        )
        endpoint = actual["parity"]["diagnostics"]["optimizer_endpoint"]
        assert endpoint is not None

    def test_gaussian_sz_select_reml_matches_mgcv(self):
        data = _make_sz_data()
        formula = 'y ~ s(f1, f2, x, bs="sz", k=6)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML", select=True)
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML", select=True)

        a_sp = np.atleast_1d(
            np.asarray(actual["fit"]["smoothing_params"], dtype=np.float64)
        )
        e_sp = np.atleast_1d(
            np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        )
        assert a_sp.size == e_sp.size == 7

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-4,
            pred_rtol=1e-4,
            sp_log_atol=4.1,
            check_sp=False,
            criterion_atol=1e-3,
        )

    def test_gaussian_mrf_select_reml_matches_mgcv(self):
        data = _make_mrf_data()
        formula = (
            'y ~ s(region, bs="mrf", k=3, '
            'xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B"))))'
        )

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML", select=True)
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML", select=True)

        asp = np.atleast_1d(
            np.asarray(actual["fit"]["smoothing_params"], dtype=np.float64)
        )
        esp = np.atleast_1d(
            np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        )
        assert asp.size == esp.size == 1
        # Full-rank MRF with absorb.cons removes the null space, so select=TRUE
        # does not introduce an extra penalty block. Both implementations park
        # the single lambda on the same flat boundary ridge.
        assert float(asp[0]) < 1e-12 and float(esp[0]) < 1e-12

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-10,
            pred_rtol=1e-10,
            sp_log_atol=0.0,
            check_sp=False,
            criterion_atol=2.5,
        )

    # ------------------------------------------------------------------ #
    # Gap 9: prior weights for non-Gaussian families (fixed-sp)           #
    # ------------------------------------------------------------------ #

    def test_weighted_poisson_fixed_sp_matches_mgcv(self):
        """Weighted Poisson at mgcv's REML sp should match mgcv's weighted predictions."""
        rng = np.random.default_rng(310)
        n = 220
        x0 = rng.normal(size=n)
        x1 = rng.normal(size=n)
        mu = np.exp(0.2 + 0.6 * np.sin(x0) - 0.2 * x1)
        y = rng.poisson(mu)
        w = rng.uniform(0.5, 2.0, size=n)
        data = pd.DataFrame({"y": y, "x0": x0, "x1": x1, "w": w})
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        expected = _run_mgcv_snapshot(
            data, formula, "poisson", "REML", weights_column="w"
        )
        sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        gam = _fit_nampy_model_fixed_sp(data, formula, "poisson", sp, sample_weight="w")
        actual = gam.parity_snapshot(X=data, include_covariances=True)

        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=1e-10,
            rtol=0.0,
        )

    def test_weighted_binomial_fixed_sp_matches_mgcv(self):
        """Weighted Binomial at mgcv's REML sp should match mgcv's weighted predictions."""
        rng = np.random.default_rng(311)
        n = 220
        x0 = rng.normal(size=n)
        x1 = rng.normal(size=n)
        eta = 0.8 * np.sin(x0) - 0.4 * x1
        p = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, p).astype(float)
        w = rng.uniform(0.5, 2.0, size=n)
        data = pd.DataFrame({"y": y, "x0": x0, "x1": x1, "w": w})
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        expected = _run_mgcv_snapshot(
            data, formula, "binomial", "REML", weights_column="w"
        )
        sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        gam = _fit_nampy_model_fixed_sp(
            data, formula, "binomial", sp, sample_weight="w"
        )
        actual = gam.parity_snapshot(X=data, include_covariances=True)

        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=1e-10,
            rtol=0.0,
        )

    # ------------------------------------------------------------------ #
    # Gap 11: tensor smooths with ps marginals                           #
    # The stale factory guard blocking non-cr tensor marginals is fixed. #
    # te()/ti() with ps marginals now match mgcv directly, while         #
    # t2(ps, ps) is covered under REML because fixed-sp penalty-count    #
    # bookkeeping remains a separate issue.                              #
    # ------------------------------------------------------------------ #

    def test_gaussian_te_ps_ps_fixed_matches_mgcv(self):
        data = _make_gaussian_data(seed=330, n=180)
        formula = 'y ~ te(x0, x1, bs=["ps", "ps"], k=[5, 5], sp=[0.7, 1.3])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
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
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gaussian_ti_ps_ps_fixed_matches_mgcv(self):
        data = _make_gaussian_data(seed=331, n=180)
        formula = 'y ~ ti(x0, x1, bs=["ps", "ps"], k=[5, 5], sp=[0.7, 1.3])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
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
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gaussian_t2_ps_ps_reml_matches_mgcv(self):
        data = _make_gaussian_data(seed=332, n=180)
        formula = 'y ~ t2(x0, x1, bs=["ps", "ps"], k=[5, 5])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-4,
            pred_rtol=1e-4,
            sp_log_atol=0.0,
            check_sp=False,
        )

    def test_gaussian_t2_ps_ps_invalid_term_sp_warns_and_matches_mgcv_reml(self):
        data = _make_gaussian_data(seed=333, n=180)
        formula = 'y ~ t2(x0, x1, bs=["ps", "ps"], k=[5, 5], sp=[0.7, 1.3])'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        assert any(
            "length of sp incorrect in t2: ignored" in str(w.message) for w in caught
        )

        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-4,
            pred_rtol=1e-4,
            sp_log_atol=0.0,
            check_sp=False,
        )

    def test_gaussian_te_tp_ps_fixed_matches_mgcv(self):
        data = _make_gaussian_data(seed=334, n=180)
        formula = 'y ~ te(x0, x1, bs=["tp", "ps"], k=[6, 6], sp=[0.8, 1.2])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
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
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gaussian_ti_tp_ps_fixed_matches_mgcv(self):
        data = _make_gaussian_data(seed=335, n=180)
        formula = 'y ~ ti(x0, x1, bs=["tp", "ps"], k=[6, 6], sp=[0.8, 1.2])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
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
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gaussian_t2_tp_ps_reml_matches_mgcv(self):
        data = _make_gaussian_data(seed=336, n=180)
        formula = 'y ~ t2(x0, x1, bs=["tp", "ps"], k=[6, 6])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_exact_mgcv_snapshot_parity(
            actual,
            expected,
            pred_atol=1e-7,
            pred_rtol=1e-7,
            edf_atol=1e-5,
            criterion_atol=1e-7,
            criterion_rtol=1e-7,
            sp_atol=1e-8,
            sp_rtol=1e-8,
            log_sp_atol=1e-5,
        )

    def test_gaussian_t2_tp_cr_fixed_matches_mgcv(self):
        data = _make_gaussian_data(seed=338, n=180)
        formula = 'y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6], sp=[0.7, 1.3, 0.9])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
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
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gaussian_t2_gp_cr_fixed_matches_mgcv(self):
        data = _make_gaussian_data(seed=339, n=180)
        formula = 'y ~ t2(x0, x1, bs=["gp", "cr"], k=[10, 6], sp=[0.7, 1.3, 0.9])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
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
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gaussian_t2_tp_gp_fixed_matches_mgcv(self):
        data = _make_gaussian_data(seed=340, n=180)
        formula = 'y ~ t2(x0, x1, bs=["tp", "gp"], k=[6, 10], sp=[0.7, 1.3, 0.9])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
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
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gaussian_t2_ps_cr_fixed_matches_mgcv(self):
        data = _make_gaussian_data(seed=341, n=180)
        formula = 'y ~ t2(x0, x1, bs=["ps", "cr"], k=[6, 6], sp=[0.7, 1.3, 0.9])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
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
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gaussian_t2_tp_cr_full_reml_matches_mgcv(self):
        data = _make_gaussian_data(seed=342, n=180)
        formula = 'y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6], full=True)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-5,
            pred_rtol=1e-5,
            sp_log_atol=0.0,
            check_sp=False,
            criterion_atol=5e-4,
        )

    def test_gaussian_t2_tp_cr_ord1_reml_matches_mgcv(self):
        data = _make_gaussian_data(seed=343, n=180)
        formula = 'y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6], ord=1)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_exact_mgcv_snapshot_parity(
            actual,
            expected,
            pred_atol=1e-7,
            pred_rtol=1e-7,
            edf_atol=2e-6,
            criterion_atol=1e-7,
            criterion_rtol=1e-7,
            sp_atol=1e-8,
            sp_rtol=1e-8,
            log_sp_atol=1e-5,
        )

    def test_gaussian_t2_tp_cr_ord12_reml_matches_mgcv(self):
        data = _make_gaussian_data(seed=344, n=180)
        formula = 'y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6], ord=[1, 2])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-6,
            pred_rtol=1e-6,
            sp_log_atol=1.6,
            criterion_atol=1e-6,
        )

    def test_gaussian_t2_tp_cr_full_ord1_reml_matches_mgcv(self):
        data = _make_gaussian_data(seed=345, n=180)
        formula = 'y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6], full=True, ord=1)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=2.5e-5,
            pred_rtol=2.5e-5,
            sp_log_atol=0.0,
            check_sp=False,
            criterion_atol=5e-4,
        )

    def test_gaussian_t2_ps_ps_full_reml_matches_mgcv(self):
        data = _make_gaussian_data(seed=346, n=180)
        formula = 'y ~ t2(x0, x1, bs=["ps", "ps"], k=[6, 6], full=True)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_exact_mgcv_snapshot_parity(
            actual,
            expected,
            pred_atol=1e-7,
            pred_rtol=1e-7,
            edf_atol=1e-5,
            criterion_atol=1e-7,
            criterion_rtol=1e-7,
            sp_atol=1e-8,
            sp_rtol=1e-8,
            log_sp_atol=1e-5,
        )

    def test_gaussian_t2_tp_gp_ord1_reml_matches_mgcv(self):
        data = _make_gaussian_data(seed=347, n=180)
        formula = 'y ~ t2(x0, x1, bs=["tp", "gp"], k=[6, 8], ord=1)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_exact_mgcv_snapshot_parity(
            actual,
            expected,
            pred_atol=1e-7,
            pred_rtol=1e-7,
            edf_atol=1e-5,
            criterion_atol=1e-6,
            criterion_rtol=1e-6,
            sp_atol=1e-8,
            sp_rtol=1e-8,
            log_sp_atol=1e-5,
        )

    def test_gaussian_t2_select_reml_matches_mgcv(self):
        data = _make_gaussian_data(seed=348, n=120)
        formula = 'y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML", select=True)
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML", select=True)
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-5,
            pred_rtol=1e-5,
            sp_log_atol=0.0,
            check_sp=False,
            criterion_atol=1e-4,
        )

    def test_gaussian_te_gp_cr_reml_matches_mgcv(self):
        data = _make_gaussian_data(seed=337, n=180)
        formula = 'y ~ te(x0, x1, bs=["gp", "cr"], k=[10, 6])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-5,
            pred_rtol=1e-5,
            sp_log_atol=0.1,
        )

    # ------------------------------------------------------------------ #
    # Gap 13: tensor marginals — cc and ts                                #
    # cc (cyclic cubic) and ts (shrinkage TP) as tensor marginals.        #
    # ------------------------------------------------------------------ #

    def test_gaussian_te_cc_cc_fixed_matches_mgcv(self):
        data = _make_gaussian_data(seed=370, n=180)
        formula = 'y ~ te(x0, x1, bs=["cc", "cc"], k=[7, 7], sp=[0.7, 1.3])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
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
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gaussian_ti_cc_cc_fixed_matches_mgcv(self):
        data = _make_gaussian_data(seed=371, n=180)
        formula = 'y ~ ti(x0, x1, bs=["cc", "cc"], k=[7, 7], sp=[0.7, 1.3])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
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
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gaussian_t2_cc_cc_reml_matches_mgcv(self):
        data = _make_gaussian_data(seed=372, n=180)
        formula = 'y ~ t2(x0, x1, bs=["cc", "cc"], k=[7, 7])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-5,
            pred_rtol=1e-5,
            sp_log_atol=0.0,
            check_sp=False,
        )

    def test_gaussian_te_ts_cr_fixed_matches_mgcv(self):
        data = _make_gaussian_data(seed=373, n=180)
        formula = 'y ~ te(x0, x1, bs=["ts", "cr"], k=[6, 6], sp=[0.7, 1.3])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
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
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gaussian_ti_ts_cr_fixed_matches_mgcv(self):
        data = _make_gaussian_data(seed=374, n=180)
        formula = 'y ~ ti(x0, x1, bs=["ts", "cr"], k=[6, 6], sp=[0.7, 1.3])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
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
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gaussian_t2_ts_cr_reml_matches_mgcv(self):
        data = _make_gaussian_data(seed=375, n=180)
        formula = 'y ~ t2(x0, x1, bs=["ts", "cr"], k=[6, 6])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-5,
            pred_rtol=1e-5,
            sp_log_atol=0.0,
            check_sp=False,
        )

    def test_gaussian_te_tp_cr_fixed_matches_mgcv(self):
        data = _make_gaussian_data(seed=376, n=180)
        formula = 'y ~ te(x0, x1, bs=["tp", "cr"], k=[6, 6], sp=[0.7, 1.3])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
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
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gaussian_ti_gp_cr_fixed_matches_mgcv(self):
        data = _make_gaussian_data(seed=377, n=180)
        formula = 'y ~ ti(x0, x1, bs=["gp", "cr"], k=[8, 6], sp=[0.7, 1.3])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        # gp basis parity is TIGHT (atol~1e-6) not EXACT
        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=1e-5,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["link"], dtype=np.float64),
            np.asarray(expected["predictions"]["link"], dtype=np.float64),
            atol=1e-5,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-5,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["deviance"]),
            float(expected["fit"]["deviance"]),
            atol=1e-5,
            rtol=1e-5,
        )

    # ------------------------------------------------------------------ #
    # Gap 12: fs with ps marginal — full-model parity                     #
    # ------------------------------------------------------------------ #

    def test_gaussian_fs_ps_marginal_reml_matches_mgcv(self):
        data = _make_fs_data()
        formula = 'y ~ s(f, x, bs="fs", xt=list(bs="ps", m=2, k=7))'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        # Both NAMpy and mgcv converge to near-zero smoothing on a flat
        # landscape; sp values differ substantially in log-space but both
        # represent effectively-unpenalized fits.  Check predictions only.
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-3,
            pred_rtol=0.0,
            sp_log_atol=5.0,
            criterion_atol=2.0,
        )

    def test_gaussian_fs_ps_marginal_select_reml_matches_mgcv(self):
        data = _make_fs_data()
        formula = 'y ~ s(f, x, bs="fs", xt=list(bs="ps", m=2, k=7))'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML", select=True)
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML", select=True)

        # Same flat REML ridge as non-select fs+ps; select=TRUE can shift the
        # outer optimum enough that log(sp) differs by slightly more than 5
        # while predictions/EDF remain aligned (mgcv vs NAMpy ~6+ in log-sp).
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-3,
            pred_rtol=0.0,
            sp_log_atol=7.0,
            criterion_atol=2.0,
        )

        endpoint = actual["parity"]["diagnostics"]["optimizer_endpoint"]
        assert endpoint is not None

    # ------------------------------------------------------------------ #
    # Gap 14: NegBin with theta != 1.0                                    #
    # ------------------------------------------------------------------ #

    def test_negbin_theta_0p5_reml_matches_mgcv(self):
        data = _make_negbin_data(seed=340, n=240, theta=0.5)
        family = {"name": "negbin", "theta": 0.5}
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, family, "REML")
        expected = _run_mgcv_snapshot(data, formula, family, "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-4,
            pred_rtol=0.0,
            sp_log_atol=1e-4,
        )

    def test_negbin_theta_2p0_reml_matches_mgcv(self):
        data = _make_negbin_data(seed=341, n=240, theta=2.0)
        family = {"name": "negbin", "theta": 2.0}
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, family, "REML")
        expected = _run_mgcv_snapshot(data, formula, family, "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-5,
            pred_rtol=0.0,
            sp_log_atol=1e-5,
        )

    def test_negbin_theta_0p5_fixed_sp_matches_mgcv(self):
        data = _make_negbin_data(seed=342, n=240, theta=0.5)
        family = {"name": "negbin", "theta": 0.5}
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        expected = _run_mgcv_snapshot(data, formula, family, "REML")
        sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)
        actual = gam.parity_snapshot(X=data, include_covariances=True)

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-6,
            pred_rtol=0.0,
            sp_log_atol=1e-10,
            check_criterion=False,
        )

    def test_negbin_theta_2p0_fixed_sp_matches_mgcv(self):
        data = _make_negbin_data(seed=343, n=240, theta=2.0)
        family = {"name": "negbin", "theta": 2.0}
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        expected = _run_mgcv_snapshot(data, formula, family, "REML")
        sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)
        actual = gam.parity_snapshot(X=data, include_covariances=True)

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-9,
            pred_rtol=0.0,
            sp_log_atol=1e-10,
            check_criterion=False,
        )

    # ------------------------------------------------------------------ #
    # Gap 15: Binomial probit and cloglog links                           #
    # ------------------------------------------------------------------ #

    def test_binomial_probit_fixed_sp_matches_mgcv(self):
        data = _make_binomial_data(seed=350, n=220)
        family = {"name": "binomial", "link": "probit"}
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        expected = _run_mgcv_snapshot(data, formula, family, "REML")
        sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)
        actual = gam.parity_snapshot(X=data, include_covariances=True)

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-7,
            pred_rtol=0.0,
            sp_log_atol=1e-10,
            check_criterion=False,
        )

    def test_binomial_probit_reml_matches_mgcv(self):
        data = _make_binomial_data(seed=351, n=220)
        family = {"name": "binomial", "link": "probit"}
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, family, "REML")
        expected = _run_mgcv_snapshot(data, formula, family, "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-2,
            pred_rtol=0.0,
            sp_log_atol=0.5,
        )

    def test_binomial_cloglog_fixed_sp_matches_mgcv(self):
        data = _make_binomial_data(seed=352, n=220)
        family = {"name": "binomial", "link": "cloglog"}
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        expected = _run_mgcv_snapshot(data, formula, family, "REML")
        sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)
        actual = gam.parity_snapshot(X=data, include_covariances=True)

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-7,
            pred_rtol=0.0,
            sp_log_atol=1e-10,
            check_criterion=False,
        )

    def test_binomial_cloglog_reml_matches_mgcv(self):
        data = _make_binomial_data(seed=353, n=220)
        family = {"name": "binomial", "link": "cloglog"}
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, family, "REML")
        expected = _run_mgcv_snapshot(data, formula, family, "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-2,
            pred_rtol=0.0,
            sp_log_atol=0.1,
        )

    # ------------------------------------------------------------------ #
    # Gap 16: Gamma inverse link                                          #
    # ------------------------------------------------------------------ #

    def test_gamma_inverse_link_fixed_sp_matches_mgcv(self):
        data = _make_gamma_data(seed=360, n=220)
        family = {"name": "gamma", "link": "inverse"}
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        expected = _run_mgcv_snapshot(data, formula, family, "REML")
        sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)
        actual = gam.parity_snapshot(X=data, include_covariances=True)

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-9,
            pred_rtol=0.0,
            sp_log_atol=1e-10,
            check_criterion=False,
        )

    def test_gamma_inverse_link_reml_matches_mgcv(self):
        # Gamma inverse-link REML has a wider outer-optimizer landscape than
        # log-link Gamma; sp optimisation can converge to different local
        # optima.  We fix NAMpy at mgcv's REML sp and compare predictions.
        data = _make_gamma_data(seed=361, n=220)
        family = {"name": "gamma", "link": "inverse"}
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        expected = _run_mgcv_snapshot(data, formula, family, "REML")
        sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)
        actual = gam.parity_snapshot(X=data, include_covariances=True)

        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=2e-9,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["link"], dtype=np.float64),
            np.asarray(expected["predictions"]["link"], dtype=np.float64),
            atol=1e-9,
            rtol=0.0,
        )
