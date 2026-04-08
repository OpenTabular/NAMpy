"""Basis/penalty parity tests: smoothCon-level checks plus per-basis-type model fits.

Covers:
 - te, ti, t2, fs, sz, mrf, re (TestParitySnapshotAPI)
 - cc, ps, gp — smoothCon basis/penalties AND fixed-sp / REML end-to-end fits
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.basemodels.gam import GAM
from nampy.gam.basis.tensor import t2_marginal_reparameterization
from nampy.gam.design.compiler import compile_predictor_designs
from nampy.gam.formula import compile_predictor_specs_from_formula, parse_gam_formula
from nampy.gam.smooths.univariate.cubic_regression import SplineTerm1D

from mgcv_parity_utils import (
    R_SCRIPT,
    _assert_allclose_up_to_column_sign,
    _assert_basic_mgcv_parity,
    _fit_nampy_model_fixed_sp,
    _fit_nampy_snapshot,
    _make_fs_data,
    _make_gaussian_data,
    _make_mrf_data,
    _make_random_effect_data,
    _make_sz_data,
    _run_mgcv_natparam_cr,
    _run_mgcv_smoothcon_matrix,
    _run_mgcv_smoothcon_matrix_unscaled,
    _run_mgcv_smoothcon_penalties,
    _run_mgcv_snapshot,
)


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

    @pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
    def test_mgcv_snapshot_script_accepts_python_tensor_formula_syntax(self):
        data = _make_gaussian_data(n=80)
        formula = 'y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3])'

        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        assert "fit" in snap
        assert "predictions" in snap
        assert np.asarray(snap["predictions"]["response"]).shape == (len(data),)

    @pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
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

    @pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
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
        actual = [np.asarray(block.matrix, dtype=np.float64) for block in design.compiled_penalties]
        target = [np.asarray(S, dtype=np.float64) for S in expected["S"]]

        assert len(actual) == len(target) == 2
        for got, want in zip(actual, target):
            np.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-10)

    @pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
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

    @pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
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
        actual = [np.asarray(pb.matrix, dtype=np.float64) for pb in design.compiled_penalties]
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
        np.testing.assert_allclose(scales[0], 1.0, atol=5e-4, rtol=5e-4)

    @pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
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

    @pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
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
        actual = [np.asarray(pb.matrix, dtype=np.float64) for pb in design.compiled_penalties]
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

    @pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
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

    @pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
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
        actual = [np.asarray(pb.matrix, dtype=np.float64) for pb in design.compiled_penalties]
        target = [np.asarray(np.array(S), dtype=np.float64) for S in expected["S"]]

        assert len(actual) == len(target)
        for got, want in zip(actual, target):
            np.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-10)

    @pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
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
        actual = [np.asarray(pb.matrix, dtype=np.float64) for pb in design.compiled_penalties]
        target = [np.asarray(np.array(S), dtype=np.float64) for S in expected["S"]]

        assert len(actual) == len(target) == 1
        np.testing.assert_allclose(actual[0], target[0], atol=1e-10, rtol=1e-10)

    @pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
    def test_mrf_smoothcon_basis_matches_mgcv(self):
        data = _make_mrf_data()
        smooth_expr_r = 's(region, bs="mrf", xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B"))))'

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

    @pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
    def test_mrf_smoothcon_penalty_matches_mgcv(self):
        data = _make_mrf_data()
        smooth_expr_r = 's(region, bs="mrf", xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B"))))'

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
        actual = [np.asarray(pb.matrix, dtype=np.float64) for pb in design.compiled_penalties]
        target = [np.asarray(np.array(S), dtype=np.float64) for S in expected["S"]]

        assert len(actual) == len(target)
        for got, want in zip(actual, target):
            np.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-10)

    @pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
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

    @pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
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

    @pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
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

    @pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
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
        actual = [np.asarray(block.matrix, dtype=np.float64) for block in design.compiled_penalties]
        target = [np.asarray(S, dtype=np.float64) for S in expected["S"]]

        assert len(actual) == len(target) == 2
        for got, want in zip(actual, target):
            np.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-10)

    @pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
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

    @pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
    def test_t2_marginal_natparam_matches_mgcv_exactly(self):
        rng = np.random.default_rng(31)
        data = pd.DataFrame({"x": rng.uniform(-2.0, 2.0, size=120)})

        term = SplineTerm1D(feature="x", k=5, basis="cr")
        term.fit(data[["x"]].to_numpy(dtype=np.float64), ["x"])
        expected = _run_mgcv_natparam_cr(data, k=5)
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

    @pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
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

        actual = [np.asarray(block.matrix, dtype=np.float64) for block in design.compiled_penalties]
        target = [np.asarray(S, dtype=np.float64) for _, S in expected["S"].items()]

        assert len(actual) == len(target) == 3
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
        term = design.compiled_terms[0].smooth.runtime

        np.testing.assert_allclose(
            np.asarray(term.basis_train, dtype=np.float64),
            np.asarray(
                term.transform_new(data[["x0", "x1"]].to_numpy(dtype=np.float64)),
                dtype=np.float64,
            ),
            atol=1e-12,
            rtol=1e-12,
        )


# ---------------------------------------------------------------------------
# Cyclic cubic spline (cc)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
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
        actual = [np.asarray(pb.matrix, dtype=np.float64) for pb in design.compiled_penalties]
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
            atol=1e-10, rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_gaussian_cc_reml_matches_mgcv(self):
        data = self._make_cyclic_data(seed=79, n=200)
        formula = 'y ~ s(x, bs="cc", k=10)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        _assert_basic_mgcv_parity(
            actual, expected,
            pred_atol=5e-3, pred_rtol=0.0,
            sp_log_atol=0.6,
        )


# ---------------------------------------------------------------------------
# P-spline (ps)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
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
        actual = [np.asarray(pb.matrix, dtype=np.float64) for pb in design.compiled_penalties]
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
            atol=1e-10, rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_gaussian_ps_reml_matches_mgcv(self):
        data = self._make_ps_data(seed=83, n=200)
        formula = 'y ~ s(x, bs="ps", k=14)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        _assert_basic_mgcv_parity(
            actual, expected,
            pred_atol=5e-3, pred_rtol=0.0,
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
            actual, expected,
            pred_atol=8e-3, pred_rtol=0.0,
            sp_log_atol=0.6,
        )


# ---------------------------------------------------------------------------
# Gaussian process smooth (gp)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
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
        _assert_allclose_up_to_column_sign(actual_X, expected_X, atol=1e-8, rtol=1e-8)

    def test_gaussian_gp_fixed_sp_matches_mgcv(self):
        data = self._make_gp_data(seed=92)
        formula = 'y ~ s(x, bs="gp", k=10, sp=1.0)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=1e-8, rtol=1e-8,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-8, rtol=1e-8,
        )

    def test_gaussian_gp_reml_matches_mgcv(self):
        data = self._make_gp_data(seed=93, n=180)
        formula = 'y ~ s(x, bs="gp", k=12)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        _assert_basic_mgcv_parity(
            actual, expected,
            pred_atol=8e-3, pred_rtol=0.0,
            sp_log_atol=0.7,
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
            actual, expected,
            pred_atol=1e-2, pred_rtol=0.0,
            sp_log_atol=0.8,
        )
