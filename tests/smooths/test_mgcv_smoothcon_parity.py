"""Basis/penalty parity tests: smoothCon-level checks plus per-basis-type model fits.

Covers:
 - te, ti, fs, sz, re (TestParitySnapshotAPI)
 - cc, ps — smoothCon basis/penalties AND fixed-sp / REML end-to-end fits
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nampy.gam import GAM
from nampy.gam.compiler.compile_predictors import compile_predictors
from nampy.gam.formula import extract_formula_terms, parse_gam_formula
from nampy.gam.linalg import matrix_self_gram
from nampy.gam.linalg import symmetric_spectrum as penalty_spectrum
from nampy.gam.smooths.univariate.bs import DerivativeBSplineTerm1D
from nampy.gam.smooths.univariate.cr import CubicSplineTerm
from nampy.gam.specs.build import build_formula_model
from tests._mgcv_snapshot_parity_shared import (
    TestPSplineSmooth as _SharedTestPSplineSmooth,
)
from tests.mgcv_invariant_policy import penalized_response_operator
from tests.mgcv_parity_utils import (
    _assert_allclose_up_to_column_sign,
    _assert_basic_mgcv_parity,
    _fit_nampy_snapshot,
    _make_fs_data,
    _make_gaussian_data,
    _make_random_effect_data,
    _make_sz_data,
    _run_mgcv_smoothcon_matrix,
    _run_mgcv_smoothcon_matrix_unscaled,
    _run_mgcv_smoothcon_penalties,
    _run_mgcv_smoothcon_predict_matrix,
    _run_mgcv_snapshot,
)


def _sym_rank(S: np.ndarray) -> int:
    S = np.asarray(S, dtype=np.float64)
    if S.size == 0:
        return 0
    ev = np.linalg.eigvalsh(0.5 * (S + S.T))
    tol = np.finfo(np.float64).eps ** 0.8 * max(float(np.max(np.abs(ev))), 1.0)
    return int(np.sum(ev > tol))


def _first_penalty(penalties):
    if isinstance(penalties, dict):
        return next(iter(penalties.values()))
    return penalties[0]


def _assert_sz_penalty_invariants(
    actual_design: np.ndarray,
    expected_design: np.ndarray,
    actual_penalties: list[np.ndarray],
    expected_penalties: list[np.ndarray],
) -> None:
    """Compare SZ penalties without depending on TP eigenvector orientation."""
    assert len(actual_penalties) == len(expected_penalties)
    for got, want in zip(actual_penalties, expected_penalties, strict=True):
        np.testing.assert_allclose(
            penalty_spectrum(got),
            penalty_spectrum(want),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            penalized_response_operator(actual_design, [got]),
            penalized_response_operator(expected_design, [want]),
            atol=1e-10,
            rtol=1e-10,
        )

    np.testing.assert_allclose(
        penalized_response_operator(actual_design, actual_penalties),
        penalized_response_operator(expected_design, expected_penalties),
        atol=1e-10,
        rtol=1e-10,
    )


def _compile_formula_design(data, formula, **build_kwargs):
    parsed = parse_gam_formula(formula)
    extracted = extract_formula_terms(parsed)
    built = build_formula_model(
        extracted, data=data, y=np.zeros(len(data)), **build_kwargs
    )
    return compile_predictors(built.X, built.feature_names, built.predictor_specs)[0]


class TestParitySnapshotAPI:
    """
    smoothCon and snapshot parity checks for basis construction, penalties, and
    representative end-to-end fits.
    """

    def test_parity_snapshot_supports_direct_gam_object(self):
        """Verify that parity snapshot supports direct gam object."""
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
        """Verify that cr raw basis reproduces constant and linear functions exactly."""
        rng = np.random.default_rng(31)
        data = pd.DataFrame({"x": rng.uniform(-2.0, 2.0, size=120)})

        term = CubicSplineTerm(feature="x", k=5, basis="cr")
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
        """Verify that mgcv snapshot script accepts python tensor formula syntax."""
        data = _make_gaussian_data(n=80)
        formula = 'y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3])'

        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        assert "fit" in snap
        assert "predictions" in snap
        assert np.asarray(snap["predictions"]["response"]).shape == (len(data),)

    def test_te_smoothcon_basis_matches_mgcv(self):
        """Verify that te smoothcon basis matches mgcv."""
        data = _make_gaussian_data(seed=7, n=80)
        smooth_expr_r = 'te(x0, x1, bs=c("cr", "cr"), k=c(5, 5), sp=c(0.7, 1.3))'

        design = _compile_formula_design(
            data,
            'y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3])',
        )

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        _assert_allclose_up_to_column_sign(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_te_runtime_penalties_match_mgcv_scaled_smoothcon(self):
        """Verify that te runtime penalties match mgcv scaled smoothcon."""
        data = _make_gaussian_data(seed=7, n=80)
        smooth_expr_r = 'te(x0, x1, bs=c("cr", "cr"), k=c(5, 5), sp=c(0.7, 1.3))'

        design = _compile_formula_design(
            data,
            'y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3])',
        )

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
        for got, want in zip(actual, target, strict=True):
            np.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-10)

    def test_te_pc_smoothcon_basis_and_penalties_match_mgcv(self):
        """te pc= uses the upstream point-evaluation constraint coordinates."""
        data = _make_gaussian_data(seed=17, n=80)
        smooth_expr_r = (
            'te(x0, x1, bs=c("cr", "cr"), k=c(5, 5), '
            'pc=c(0.2, -0.3), sp=c(0.7, 1.3))'
        )
        design = _compile_formula_design(
            data,
            'y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5], '
            'pc=[0.2, -0.3], sp=[0.7, 1.3])',
        )
        expected_x = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)
        expected_s = _run_mgcv_smoothcon_penalties(
            data, smooth_expr_r, absorb_cons=True, scale_penalty=True
        )

        np.testing.assert_allclose(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected_x["X"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        actual = [np.asarray(p.matrix) for p in design.compiled_penalties]
        target = [np.asarray(S) for S in expected_s["S"]]
        assert len(actual) == len(target) == 2
        for got, want in zip(actual, target, strict=True):
            np.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-10)

    def test_te_ps_nested_margin_orders_match_scalar_margin_orders(self):
        """Verify that te ps nested margin orders match scalar margin orders."""
        data = _make_gaussian_data(seed=19, n=80)

        design_nested = _compile_formula_design(
            data,
            'y ~ te(x0, x1, bs=["ps", "ps"], k=[6, 7], m=[[1, 1], [3, 3]], sp=[0.7, 1.3])',
        )
        design_scalar = _compile_formula_design(
            data,
            'y ~ te(x0, x1, bs=["ps", "ps"], k=[6, 7], m=[1, 3], sp=[0.7, 1.3])',
        )

        _assert_allclose_up_to_column_sign(
            np.asarray(design_nested.design_matrix, dtype=np.float64),
            np.asarray(design_scalar.design_matrix, dtype=np.float64),
            atol=1e-12,
            rtol=1e-12,
        )

        nested_penalties = [
            np.asarray(block.matrix, dtype=np.float64)
            for block in design_nested.compiled_penalties
        ]
        scalar_penalties = [
            np.asarray(block.matrix, dtype=np.float64)
            for block in design_scalar.compiled_penalties
        ]
        assert len(nested_penalties) == len(scalar_penalties) == 2
        for got, want in zip(nested_penalties, scalar_penalties, strict=True):
            np.testing.assert_allclose(got, want, atol=1e-12, rtol=1e-12)

    def test_te_ps_margin_orders_basis_matches_mgcv(self):
        """Verify that te ps margin orders basis matches mgcv."""
        data = _make_gaussian_data(seed=20, n=80)
        smooth_expr_r = (
            'te(x0, x1, bs=c("ps", "ps"), k=c(6, 7), m=c(1, 3), sp=c(0.7, 1.3))'
        )

        design = _compile_formula_design(
            data,
            'y ~ te(x0, x1, bs=["ps", "ps"], k=[6, 7], m=[1, 3], sp=[0.7, 1.3])',
        )

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        _assert_allclose_up_to_column_sign(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_te_ps_margin_orders_penalties_match_mgcv(self):
        """Verify that te ps margin orders penalties match mgcv."""
        data = _make_gaussian_data(seed=20, n=80)
        smooth_expr_r = (
            'te(x0, x1, bs=c("ps", "ps"), k=c(6, 7), m=c(1, 3), sp=c(0.7, 1.3))'
        )

        design = _compile_formula_design(
            data,
            'y ~ te(x0, x1, bs=["ps", "ps"], k=[6, 7], m=[1, 3], sp=[0.7, 1.3])',
        )

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
        for got, want in zip(actual, target, strict=True):
            np.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-10)

    def test_fs_smoothcon_basis_matches_mgcv(self):
        """Verify that fs smoothcon basis matches mgcv."""
        data = _make_fs_data()
        smooth_expr_r = 's(f, x, bs="fs")'

        design = _compile_formula_design(data, 'y ~ s(f, x, bs="fs")')

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        actual_basis = np.asarray(design.design_matrix, dtype=np.float64)
        expected_basis = np.asarray(expected["X"], dtype=np.float64)
        np.testing.assert_allclose(
            actual_basis @ actual_basis.T,
            expected_basis @ expected_basis.T,
            atol=1e-10,
            rtol=1e-10,
        )

    def test_fs_smoothcon_penalties_match_mgcv(self):
        """Verify that fs smoothcon penalties match mgcv."""
        data = _make_fs_data()
        smooth_expr_r = 's(f, x, bs="fs")'

        design = _compile_formula_design(data, 'y ~ s(f, x, bs="fs")')

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
        for got, want in zip(actual, target, strict=True):
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

    def test_fs_smoothcon_ps_basis_matches_mgcv(self):
        """Verify that fs smoothcon ps basis matches mgcv."""
        data = _make_fs_data()
        smooth_expr_r = 's(f, x, bs="fs", xt=list(bs="ps", m=2, k=7))'

        design = _compile_formula_design(
            data, 'y ~ s(f, x, bs="fs", xt=list(bs="ps", m=2, k=7))'
        )

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        actual_basis = np.asarray(design.design_matrix, dtype=np.float64)
        expected_basis = np.asarray(expected["X"], dtype=np.float64)
        np.testing.assert_allclose(
            actual_basis @ actual_basis.T,
            expected_basis @ expected_basis.T,
            atol=1e-10,
            rtol=1e-10,
        )

    def test_fs_smoothcon_ps_penalties_match_mgcv(self):
        """Verify that fs smoothcon ps penalties match mgcv."""
        data = _make_fs_data()
        smooth_expr_r = 's(f, x, bs="fs", xt=list(bs="ps", m=2, k=7))'

        design = _compile_formula_design(
            data, 'y ~ s(f, x, bs="fs", xt=list(bs="ps", m=2, k=7))'
        )

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
        for got, want in zip(actual, target, strict=True):
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
        """Verify that sz smoothcon basis matches mgcv."""
        data = _make_sz_data()
        smooth_expr_r = 's(f1, f2, x, bs="sz", k=6)'

        design = _compile_formula_design(data, 'y ~ s(f1, f2, x, bs="sz", k=6)')

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        np.testing.assert_allclose(
            matrix_self_gram(design.design_matrix),
            matrix_self_gram(expected["X"]),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_sz_smoothcon_penalties_match_mgcv(self):
        """Verify that sz smoothcon penalties match mgcv."""
        data = _make_sz_data()
        smooth_expr_r = 's(f1, f2, x, bs="sz", k=6)'

        design = _compile_formula_design(data, 'y ~ s(f1, f2, x, bs="sz", k=6)')

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
        expected_design = np.asarray(
            _run_mgcv_smoothcon_matrix(data, smooth_expr_r)["X"],
            dtype=np.float64,
        )

        _assert_sz_penalty_invariants(
            np.asarray(design.design_matrix, dtype=np.float64),
            expected_design,
            actual,
            target,
        )

    def test_sz_ps_smoothcon_penalties_match_mgcv(self):
        """Verify that sz with a ps base uses mgcv's raw-base penalty scaling."""
        data = _make_sz_data()
        smooth_expr_r = 's(f1, x, bs="sz", k=7, m=2, xt=list(bs="ps"))'

        design = _compile_formula_design(
            data, 'y ~ s(f1, x, bs="sz", k=7, m=2, xt=list(bs="ps"))'
        )

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
        for got, want in zip(actual, target, strict=True):
            np.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-10)

    def test_sz_smoothcon_shared_id_penalty_matches_mgcv(self):
        """Verify that sz smoothcon shared id penalty matches mgcv."""
        data = _make_sz_data()
        smooth_expr_r = 's(f1, f2, x, bs="sz", k=6, id="shared")'

        design = _compile_formula_design(
            data, 'y ~ s(f1, f2, x, bs="sz", k=6, id="shared")'
        )

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
        expected_design = np.asarray(
            _run_mgcv_smoothcon_matrix(data, smooth_expr_r)["X"],
            dtype=np.float64,
        )

        assert len(actual) == len(target) == 1
        _assert_sz_penalty_invariants(
            np.asarray(design.design_matrix, dtype=np.float64),
            expected_design,
            actual,
            target,
        )

    def test_re_smoothcon_factor_basis_matches_mgcv(self):
        """Verify that re smoothcon factor basis matches mgcv."""
        data = _make_random_effect_data()
        smooth_expr_r = 's(f, bs="re")'

        design = _compile_formula_design(data, 'y ~ s(f, bs="re")')

        expected = _run_mgcv_smoothcon_matrix_unscaled(data[["f"]], smooth_expr_r)

        np.testing.assert_allclose(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_re_smoothcon_numeric_factor_basis_matches_mgcv(self):
        """Verify that re smoothcon numeric factor basis matches mgcv."""
        data = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "f": ["b", "a", "c", "a"]})
        smooth_expr_r = 's(x, f, bs="re")'

        design = _compile_formula_design(data, 'y ~ s(x, f, bs="re")')

        expected = _run_mgcv_smoothcon_matrix_unscaled(data, smooth_expr_r)

        np.testing.assert_allclose(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_ti_smoothcon_basis_matches_mgcv(self):
        """Verify that ti smoothcon basis matches mgcv."""
        data = _make_gaussian_data(seed=13, n=80)
        smooth_expr_r = 'ti(x0, x1, bs=c("cr", "cr"), k=c(5, 5), sp=c(0.7, 1.3))'

        design = _compile_formula_design(
            data, 'y ~ ti(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3])'
        )

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        np.testing.assert_allclose(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_ti_runtime_penalties_match_mgcv_scaled_smoothcon(self):
        """Verify that ti runtime penalties match mgcv scaled smoothcon."""
        data = _make_gaussian_data(seed=13, n=80)
        smooth_expr_r = 'ti(x0, x1, bs=c("cr", "cr"), k=c(5, 5), sp=c(0.7, 1.3))'

        design = _compile_formula_design(
            data, 'y ~ ti(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3])'
        )

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
        for got, want in zip(actual, target, strict=True):
            np.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-10)

    def test_ti_pc_smoothcon_basis_and_penalties_match_mgcv(self):
        """ti pc= is applied after marginal centering and tensor construction."""
        data = _make_gaussian_data(seed=21, n=80)
        smooth_expr_r = (
            'ti(x0, x1, bs=c("cr", "ps"), k=c(5, 6), '
            'pc=c(0.2, -0.3), sp=c(0.7, 1.3))'
        )
        design = _compile_formula_design(
            data,
            'y ~ ti(x0, x1, bs=["cr", "ps"], k=[5, 6], '
            'pc=[0.2, -0.3], sp=[0.7, 1.3])',
        )
        expected_x = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)
        expected_s = _run_mgcv_smoothcon_penalties(
            data, smooth_expr_r, absorb_cons=True, scale_penalty=True
        )

        np.testing.assert_allclose(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected_x["X"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        actual = [np.asarray(p.matrix) for p in design.compiled_penalties]
        target = [np.asarray(S) for S in expected_s["S"]]
        assert len(actual) == len(target) == 2
        for got, want in zip(actual, target, strict=True):
            np.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-10)

    def test_ti_ps_margin_orders_basis_matches_mgcv(self):
        """Verify that ti ps margin orders basis matches mgcv."""
        data = _make_gaussian_data(seed=21, n=80)
        smooth_expr_r = (
            'ti(x0, x1, bs=c("ps", "ps"), k=c(6, 7), m=c(1, 3), sp=c(0.7, 1.3))'
        )

        design = _compile_formula_design(
            data,
            'y ~ ti(x0, x1, bs=["ps", "ps"], k=[6, 7], m=[1, 3], sp=[0.7, 1.3])',
        )

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        np.testing.assert_allclose(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_ti_ps_margin_orders_penalties_match_mgcv(self):
        """Verify that ti ps margin orders penalties match mgcv."""
        data = _make_gaussian_data(seed=21, n=80)
        smooth_expr_r = (
            'ti(x0, x1, bs=c("ps", "ps"), k=c(6, 7), m=c(1, 3), sp=c(0.7, 1.3))'
        )

        design = _compile_formula_design(
            data,
            'y ~ ti(x0, x1, bs=["ps", "ps"], k=[6, 7], m=[1, 3], sp=[0.7, 1.3])',
        )

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
        for got, want in zip(actual, target, strict=True):
            np.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-10)

# ---------------------------------------------------------------------------
# Cyclic cubic spline (cc)
# ---------------------------------------------------------------------------


class TestCyclicCubicSmooth:
    """Cyclic cubic regression spline (bs='cc') parity against mgcv."""

    def _make_cyclic_data(self, seed=77, n=180):
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.0, 2 * np.pi, size=n)
        y = np.sin(x) + 0.3 * np.cos(2 * x) + rng.normal(scale=0.12, size=n)
        return pd.DataFrame({"y": y, "x": x})

    def test_cc_smoothcon_basis_matches_mgcv(self):
        """Verify that cc smoothcon basis matches mgcv."""
        data = self._make_cyclic_data()
        smooth_expr_r = 's(x, bs="cc", k=9)'

        design = _compile_formula_design(data, 'y ~ s(x, bs="cc", k=9)')

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        _assert_allclose_up_to_column_sign(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_cc_smoothcon_penalties_match_mgcv(self):
        """Verify that cc smoothcon penalties match mgcv."""
        data = self._make_cyclic_data()
        smooth_expr_r = 's(x, bs="cc", k=9)'

        design = _compile_formula_design(data, 'y ~ s(x, bs="cc", k=9)')

        expected = _run_mgcv_smoothcon_penalties(
            data, smooth_expr_r, absorb_cons=True, scale_penalty=True
        )
        actual = [
            np.asarray(pb.matrix, dtype=np.float64) for pb in design.compiled_penalties
        ]
        target = [np.asarray(S, dtype=np.float64) for S in expected["S"]]

        assert len(actual) == len(target) == 1
        np.testing.assert_allclose(actual[0], target[0], atol=1e-10, rtol=1e-10)

    def test_cc_pc_smoothcon_basis_matches_mgcv(self):
        """Verify that cc pc smoothcon basis matches mgcv."""
        data = self._make_cyclic_data()
        smooth_expr_r = 's(x, bs="cc", k=8, pc=0.5)'

        design = _compile_formula_design(
            data,
            'y ~ s(x, bs="cc", k=8, pc=0.5)',
        )

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        np.testing.assert_allclose(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_cc_pc_smoothcon_penalties_match_mgcv(self):
        """Verify that cc pc smoothcon penalties match mgcv."""
        data = self._make_cyclic_data()
        smooth_expr_r = 's(x, bs="cc", k=8, pc=0.5)'

        design = _compile_formula_design(
            data,
            'y ~ s(x, bs="cc", k=8, pc=0.5)',
        )

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
        """Verify that gaussian cc fixed sp matches mgcv exactly."""
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
        """Verify that gaussian cc REML matches mgcv."""
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


# ---------------------------------------------------------------------------
# P-spline (ps)
# ---------------------------------------------------------------------------


class TestCyclicPSplineSmooth:
    """Cyclic P-spline (bs='cp') constructor and fit parity."""

    @staticmethod
    def _make_data(seed=181, n=190):
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.0, 2.0 * np.pi, size=n)
        y = np.sin(x) + 0.25 * np.cos(2.0 * x) + rng.normal(scale=0.12, size=n)
        return pd.DataFrame({"y": y, "x": x})

    def test_cp_smoothcon_basis_and_penalty_match_mgcv(self):
        data = self._make_data()
        formula = 'y ~ s(x, bs="cp", k=11, m=[2,1])'
        smooth_expr = 's(x, bs="cp", k=11, m=c(2,1))'
        design = _compile_formula_design(data, formula)
        expected_basis = _run_mgcv_smoothcon_matrix(data, smooth_expr)
        expected_penalties = _run_mgcv_smoothcon_penalties(
            data, smooth_expr, absorb_cons=True, scale_penalty=True
        )

        np.testing.assert_allclose(
            design.design_matrix, expected_basis["X"], atol=1e-10, rtol=1e-10
        )
        actual = [pb.matrix for pb in design.compiled_penalties]
        assert len(actual) == len(expected_penalties["S"]) == 1
        np.testing.assert_allclose(
            actual[0], expected_penalties["S"][0], atol=1e-10, rtol=1e-10
        )

    def test_cp_point_constraint_matches_mgcv(self):
        data = self._make_data(seed=182)
        formula = 'y ~ s(x, bs="cp", k=9, pc=0.0, sp=0.6)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        np.testing.assert_allclose(
            actual["predictions"]["response"],
            expected["predictions"]["response"],
            atol=2e-10,
            rtol=2e-10,
        )

    def test_cp_fixed_sp_fit_matches_mgcv(self):
        data = self._make_data(seed=183)
        formula = 'y ~ s(x, bs="cp", k=10, m=[2,2], sp=0.7)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        np.testing.assert_allclose(
            actual["predictions"]["response"],
            expected["predictions"]["response"],
            atol=2e-10,
            rtol=2e-10,
        )
        np.testing.assert_allclose(
            actual["fit"]["cov_bayes"],
            expected["fit"]["cov_bayes"],
            atol=2e-10,
            rtol=2e-10,
        )

    def test_cp_reml_fit_matches_mgcv(self):
        data = self._make_data(seed=184, n=210)
        formula = 'y ~ s(x, bs="cp", k=11, m=[2,2])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=2e-9,
            pred_rtol=2e-9,
            sp_log_atol=2e-8,
            criterion_atol=2e-8,
        )


class TestDerivativeBSplineSmooth:
    """Integrated-derivative B-spline (bs='bs') constructor and fit parity."""

    @staticmethod
    def _make_data(seed=191, n=190):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-2.0, 2.0, size=n)
        y = np.sin(1.3 * x) + 0.2 * x**2 + rng.normal(scale=0.12, size=n)
        return pd.DataFrame({"y": y, "x": x})

    def test_bs_smoothcon_basis_and_multiple_penalties_match_mgcv(self):
        data = self._make_data()
        formula = 'y ~ s(x, bs="bs", k=10, m=[3,2,1,0])'
        smooth_expr = 's(x, bs="bs", k=10, m=c(3,2,1,0))'
        design = _compile_formula_design(data, formula)
        expected_basis = _run_mgcv_smoothcon_matrix(data, smooth_expr)
        expected_penalties = _run_mgcv_smoothcon_penalties(
            data, smooth_expr, absorb_cons=True, scale_penalty=True
        )

        np.testing.assert_allclose(
            design.design_matrix, expected_basis["X"], atol=2e-10, rtol=2e-10
        )
        actual = [pb.matrix for pb in design.compiled_penalties]
        assert len(actual) == len(expected_penalties["S"]) == 3
        for actual_penalty, expected_penalty in zip(
            actual, expected_penalties["S"], strict=True
        ):
            np.testing.assert_allclose(
                actual_penalty, expected_penalty, atol=2e-10, rtol=2e-10
            )

    def test_bs_point_constraint_matches_mgcv(self):
        data = self._make_data(seed=192)
        formula = 'y ~ s(x, bs="bs", k=9, pc=0.0, sp=0.6)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        np.testing.assert_allclose(
            actual["predictions"]["response"],
            expected["predictions"]["response"],
            atol=2e-9,
            rtol=2e-9,
        )

    def test_bs_multiple_fixed_sp_fit_matches_mgcv(self):
        data = self._make_data(seed=193)
        formula = 'y ~ s(x, bs="bs", k=10, m=[3,2,0], sp=[0.7,0.9])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        np.testing.assert_allclose(
            actual["predictions"]["response"],
            expected["predictions"]["response"],
            atol=2e-9,
            rtol=2e-9,
        )
        np.testing.assert_allclose(
            actual["fit"]["cov_bayes"],
            expected["fit"]["cov_bayes"],
            atol=2e-9,
            rtol=2e-9,
        )

    def test_bs_reml_fit_matches_mgcv(self):
        data = self._make_data(seed=194, n=210)
        formula = 'y ~ s(x, bs="bs", k=11, m=[3,2])'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=3e-8,
            pred_rtol=3e-8,
            sp_log_atol=3e-7,
            criterion_atol=3e-8,
        )

    def test_bs_select_reml_matches_mgcv(self):
        data = self._make_data(seed=195, n=210)
        formula = 'y ~ s(x, bs="bs", k=10)'
        actual = _fit_nampy_snapshot(
            data, formula, "gaussian", "REML", select=True
        )
        expected = _run_mgcv_snapshot(
            data, formula, "gaussian", "REML", select=True
        )
        assert len(actual["fit"]["smoothing_params"]) == 2
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=3e-8,
            pred_rtol=3e-8,
            sp_log_atol=4e-7,
            criterion_atol=3e-8,
        )

    def test_bs_derivative_prediction_and_linear_tails_match_mgcv(self):
        data = self._make_data(seed=196)
        newdata = pd.DataFrame({"x": [-3.0, -1.0, 0.5, 3.0]})
        term = DerivativeBSplineTerm1D(feature="x", k=10, m=(3, 2))
        term.fit(data[["x"]].to_numpy(dtype=np.float64), ["x"])
        for order in (1, 2):
            actual = term.derivative_matrix(
                newdata[["x"]].to_numpy(dtype=np.float64), order=order
            )
            expected = _run_mgcv_smoothcon_predict_matrix(
                data,
                newdata,
                's(x, bs="bs", k=10, m=c(3,2))',
                deriv=order,
            )
            np.testing.assert_allclose(
                actual, expected["X"], atol=2e-10, rtol=2e-10
            )

    def test_bs_four_knot_prediction_interval_matches_mgcv(self):
        data = self._make_data(seed=197)
        knots = {"x": [-3.0, -2.2, 2.2, 3.0]}
        newdata = pd.DataFrame({"x": [-4.0, -2.7, 0.0, 2.7, 4.0]})
        term = DerivativeBSplineTerm1D(
            feature="x", k=10, m=(3, 1), knots=knots["x"]
        )
        term.fit(data[["x"]].to_numpy(dtype=np.float64), ["x"])
        actual = term.transform_new(newdata[["x"]].to_numpy(dtype=np.float64))
        expected = _run_mgcv_smoothcon_predict_matrix(
            data,
            newdata,
            's(x, bs="bs", k=10, m=c(3,1))',
            knots=knots,
        )
        np.testing.assert_allclose(
            actual, expected["X"], atol=2e-10, rtol=2e-10
        )


class TestDuchonSplineSmooth:
    """Duchon regression-spline smoothCon parity against mgcv."""

    @staticmethod
    def _make_data(seed=198, n=130):
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(-2.0, 2.0, size=n)
        x1 = rng.uniform(-1.5, 1.5, size=n)
        y = np.sin(1.2 * x0) + 0.3 * x1**2 + rng.normal(scale=0.12, size=n)
        return pd.DataFrame({"y": y, "x0": x0, "x1": x1})

    def test_ds_1d_smoothcon_basis_and_penalty_match_mgcv(self):
        data = self._make_data(seed=198)
        formula = 'y ~ s(x0, bs="ds", k=11, sp=.7)'
        expression = 's(x0, bs="ds", k=11, sp=.7)'
        design = _compile_formula_design(data, formula)
        expected_x = _run_mgcv_smoothcon_matrix(data, expression)
        expected_s = _run_mgcv_smoothcon_penalties(
            data, expression, absorb_cons=True, scale_penalty=True
        )
        actual_x = np.asarray(design.design_matrix, dtype=np.float64)
        target_x = np.asarray(expected_x["X"], dtype=np.float64)
        actual_s = np.asarray(design.compiled_penalties[0].matrix, dtype=np.float64)
        target_s = np.asarray(_first_penalty(expected_s["S"]), dtype=np.float64)

        _assert_allclose_up_to_column_sign(actual_x, target_x, atol=2e-8, rtol=2e-8)
        np.testing.assert_allclose(
            penalty_spectrum(actual_s),
            penalty_spectrum(target_s),
            atol=2e-8,
            rtol=2e-8,
        )
        np.testing.assert_allclose(
            penalized_response_operator(actual_x, [actual_s]),
            penalized_response_operator(target_x, [target_s]),
            atol=2e-8,
            rtol=2e-8,
        )

    def test_ds_2d_custom_order_smoothcon_basis_and_penalty_match_mgcv(self):
        data = self._make_data(seed=199)
        formula = 'y ~ s(x0, x1, bs="ds", k=10, m=[1,.5], sp=.7)'
        expression = 's(x0, x1, bs="ds", k=10, m=c(1,.5), sp=.7)'
        design = _compile_formula_design(data, formula)
        expected_x = _run_mgcv_smoothcon_matrix(data, expression)
        expected_s = _run_mgcv_smoothcon_penalties(
            data, expression, absorb_cons=True, scale_penalty=True
        )
        actual_x = np.asarray(design.design_matrix, dtype=np.float64)
        target_x = np.asarray(expected_x["X"], dtype=np.float64)
        actual_s = np.asarray(design.compiled_penalties[0].matrix, dtype=np.float64)
        target_s = np.asarray(_first_penalty(expected_s["S"]), dtype=np.float64)

        _assert_allclose_up_to_column_sign(actual_x, target_x, atol=2e-8, rtol=2e-8)
        np.testing.assert_allclose(
            penalty_spectrum(actual_s),
            penalty_spectrum(target_s),
            atol=2e-8,
            rtol=2e-8,
        )
        np.testing.assert_allclose(
            penalized_response_operator(actual_x, [actual_s]),
            penalized_response_operator(target_x, [target_s]),
            atol=2e-8,
            rtol=2e-8,
        )


class TestSphericalSplineSmooth:
    """Spherical-spline smoothCon parity against mgcv 1.9-4."""

    @staticmethod
    def _make_data(seed=951, n=130):
        rng = np.random.default_rng(seed)
        lo = rng.uniform(-180.0, 180.0, size=n)
        la = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))
        y = np.sin(np.deg2rad(lo)) * np.cos(np.deg2rad(la - 10.0))
        return pd.DataFrame({"y": y, "la": la, "lo": lo})

    @staticmethod
    def _assert_basis_and_penalties(data, formula, expression, *, n_penalties=1):
        design = _compile_formula_design(data, formula)
        expected_x = _run_mgcv_smoothcon_matrix(data, expression)
        actual_x = np.asarray(design.design_matrix, dtype=np.float64)
        target_x = np.asarray(expected_x["X"], dtype=np.float64)
        np.testing.assert_allclose(
            actual_x @ np.linalg.pinv(actual_x),
            target_x @ np.linalg.pinv(target_x),
            atol=2e-8,
            rtol=2e-8,
        )
        actual_s = [
            np.asarray(block.matrix, dtype=np.float64)
            for block in design.compiled_penalties
        ]
        assert len(actual_s) == n_penalties
        if n_penalties == 0:
            return
        expected_s = _run_mgcv_smoothcon_penalties(
            data, expression, absorb_cons=True, scale_penalty=True
        )
        penalty_payload = expected_s["S"]
        if isinstance(penalty_payload, dict):
            penalty_payload = list(penalty_payload.values())
        target_s = [np.asarray(S, dtype=np.float64) for S in penalty_payload]
        assert len(target_s) == n_penalties
        np.testing.assert_allclose(
            penalized_response_operator(actual_x, actual_s),
            penalized_response_operator(target_x, target_s),
            atol=2e-8,
            rtol=2e-8,
        )

    def test_sos_default_smoothcon_basis_and_penalty_match_mgcv(self):
        data = self._make_data(seed=951)
        self._assert_basis_and_penalties(
            data,
            'y ~ s(la, lo, bs="sos", k=12, sp=.7)',
            's(la, lo, bs="sos", k=12, sp=.7)',
        )

    def test_sos_duchon_tail_smoothcon_basis_and_penalty_match_mgcv(self):
        data = self._make_data(seed=952)
        self._assert_basis_and_penalties(
            data,
            'y ~ s(la, lo, bs="sos", k=12, m=-1, sp=.7)',
            's(la, lo, bs="sos", k=12, m=-1, sp=.7)',
        )

    def test_sos_pc_smoothcon_basis_and_penalty_match_mgcv(self):
        data = self._make_data(seed=953)
        self._assert_basis_and_penalties(
            data,
            'y ~ s(la, lo, bs="sos", k=12, pc=[0,0], sp=.7)',
            's(la, lo, bs="sos", k=12, pc=c(0,0), sp=.7)',
        )

    def test_sos_fixed_smoothcon_basis_matches_mgcv_without_penalty(self):
        data = self._make_data(seed=954)
        self._assert_basis_and_penalties(
            data,
            'y ~ s(la, lo, bs="sos", k=12, fx=True)',
            's(la, lo, bs="sos", k=12, fx=TRUE)',
            n_penalties=0,
        )


class TestPSplineSmooth(_SharedTestPSplineSmooth):
    """P-spline (bs='ps') standalone parity against mgcv."""

    def _make_ps_data(self, seed=81, n=180):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-2.0, 2.0, size=n)
        y = np.sin(1.3 * x) + 0.2 * x**2 + rng.normal(scale=0.14, size=n)
        return pd.DataFrame({"y": y, "x": x})

    def test_ps_smoothcon_basis_matches_mgcv(self):
        """Verify that ps smoothcon basis matches mgcv."""
        data = self._make_ps_data()
        smooth_expr_r = 's(x, bs="ps", k=12)'

        design = _compile_formula_design(data, 'y ~ s(x, bs="ps", k=12)')

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        _assert_allclose_up_to_column_sign(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_ps_smoothcon_penalties_match_mgcv(self):
        """Verify that ps smoothcon penalties match mgcv."""
        data = self._make_ps_data()
        smooth_expr_r = 's(x, bs="ps", k=12)'

        design = _compile_formula_design(data, 'y ~ s(x, bs="ps", k=12)')

        expected = _run_mgcv_smoothcon_penalties(
            data, smooth_expr_r, absorb_cons=True, scale_penalty=True
        )
        actual = [
            np.asarray(pb.matrix, dtype=np.float64) for pb in design.compiled_penalties
        ]
        target = [np.asarray(S, dtype=np.float64) for S in expected["S"]]

        assert len(actual) == len(target) == 1
        np.testing.assert_allclose(actual[0], target[0], atol=1e-10, rtol=1e-10)

    def test_ps_pc_smoothcon_basis_matches_mgcv(self):
        """Verify that ps pc smoothcon basis matches mgcv."""
        data = self._make_ps_data()
        smooth_expr_r = 's(x, bs="ps", k=8, pc=0.0)'

        design = _compile_formula_design(
            data,
            'y ~ s(x, bs="ps", k=8, pc=0.0)',
        )

        expected = _run_mgcv_smoothcon_matrix(data, smooth_expr_r)

        np.testing.assert_allclose(
            np.asarray(design.design_matrix, dtype=np.float64),
            np.asarray(expected["X"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_ps_pc_smoothcon_penalties_match_mgcv(self):
        """Verify that ps pc smoothcon penalties match mgcv."""
        data = self._make_ps_data()
        smooth_expr_r = 's(x, bs="ps", k=8, pc=0.0)'

        design = _compile_formula_design(
            data,
            'y ~ s(x, bs="ps", k=8, pc=0.0)',
        )

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
        """Verify that gaussian ps fixed sp matches mgcv exactly."""
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
        """Verify that gaussian ps REML matches mgcv."""
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
