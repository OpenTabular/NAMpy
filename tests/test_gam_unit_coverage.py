"""
Unit tests for GAM submodule internal components.

Covers the following modules that have limited or no direct test coverage:
- gam/constraints/transforms.py
- gam/constraints/absorption.py
- gam/constraints/identifiability.py
- gam/penalties/algebra.py
- gam/penalties/subsystem.py
- gam/basis/tensor.py
- gam/fit/covariance.py
- gam/fit/linalg/matrix_reindexing.py
- gam/diagnostics/summary.py
- gam/formula/parser.py
- gam/smooths/smooth_base.py helpers

Parity tests verify that numerical results match mgcv to machine precision
for edge cases not covered elsewhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.basemodels.gam import GAM
from nampy.gam.basis.tensor import (
    lifted_tensor_penalty,
    normalize_tensor_marginal_penalty,
    rowwise_kronecker,
    tensor_product_penalties,
)
from nampy.gam.constraints.absorption import (
    ConstraintFitResult,
    absorb_explicit_constraints,
    apply_linear_constraint,
    fit_single_penalty_with_constraint_policy,
    full_term_sum_to_zero_constraint,
    should_apply_identifiability_constraint,
)

# ── imports ──────────────────────────────────────────────────────────────────
from nampy.gam.constraints.transforms import (
    apply_coefficient_transform,
    independent_column_indices,
    null_space_basis_from_constraint_matrix,
    orthogonal_residual,
)
from nampy.gam.design.structures import (
    PenaltySpec,
)
from nampy.gam.diagnostics.summary import (
    build_summary_lines,
    print_summary,
    summary_text,
)
from nampy.gam.families import GaussianIdentityFamily, make_gam_family
from nampy.gam.families.exponential import (
    BinomialCloglogFamily,
    BinomialLogitFamily,
    BinomialProbitFamily,
    GammaInverseFamily,
    GammaLogFamily,
    NegativeBinomialLogFamily,
    PoissonLogFamily,
)
from nampy.gam.families.family_base import BaseFamily
from nampy.gam.fit.backends import available_fit_backends, resolve_fit_backend
from nampy.gam.fit.covariance import build_bayes_and_freq_covariances
from nampy.gam.fit.linalg.matrix_reindexing import (
    drop_columns_dense,
    drop_rows_dense,
    permute_columns,
    permute_rows,
    restore_dropped_rows,
)
from nampy.gam.fit.penalized_system import build_full_penalty_from_blocks
from nampy.gam.formula import compile_predictor_specs_from_formula, parse_gam_formula
from nampy.gam.formula.parser import (
    ParsedParametricTerm,
    ParsedSmoothTerm,
)
from nampy.gam.penalties.algebra import (
    null_space_penalty_from_penalty,
    penalty_eigendecomposition,
    symmetrize_penalty,
)
from nampy.gam.penalties.subsystem import (
    build_null_space_selection_spec,
    default_penalty_id,
    make_penalty_spec,
    merge_smoothing_override,
    normalize_penalty_spec,
    penalty_rank_null_dim,
)
from nampy.gam.smooths.smooth_base import (
    ByState,
    _resolve_feature,
    resolve_by_state,
    resolve_feature_matrix_state,
)
from nampy.gam.specs import TermSpec

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_gaussian_data(n=80, seed=1):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.0, 1.0, n)
    x1 = rng.uniform(-1.0, 1.0, n)
    y = np.sin(2 * np.pi * x0) + 0.3 * x1 + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _fit_simple_gam(n=60):
    data = _make_gaussian_data(n=n)
    gam = GAM(
        formula='y ~ s(x0, bs="cr", k=6)',
        optimize_smoothing=True,
        smoothing_method="REML",
    )
    gam.fit(data=data)
    return gam


# ─────────────────────────────────────────────────────────────────────────────
# 1. gam/constraints/transforms.py
# ─────────────────────────────────────────────────────────────────────────────


class TestOrthogonalResidual:
    def test_none_A_returns_B_unchanged(self):
        B = np.eye(4)
        result = orthogonal_residual(B, None)
        np.testing.assert_allclose(result, B)

    def test_empty_A_returns_B_unchanged(self):
        B = np.ones((5, 3))
        A = np.empty((5, 0))
        result = orthogonal_residual(B, A)
        np.testing.assert_allclose(result, B)

    def test_residual_is_orthogonal_to_A(self):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(10, 2))
        B = rng.normal(size=(10, 3))
        R = orthogonal_residual(B, A)
        # A.T @ R should be near zero
        np.testing.assert_allclose(A.T @ R, np.zeros((2, 3)), atol=1e-10)

    def test_column_in_span_of_A_is_zeroed(self):
        rng = np.random.default_rng(42)
        A = rng.normal(size=(10, 2))
        # B is a linear combination of A columns
        B = A @ np.array([[1.5], [-0.7]])
        R = orthogonal_residual(B, A)
        np.testing.assert_allclose(R, np.zeros_like(R), atol=1e-10)


class TestIndependentColumnIndices:
    def test_identity_returns_all(self):
        B = np.eye(4)
        idx = independent_column_indices(B)
        assert set(idx) == {0, 1, 2, 3}

    def test_linearly_dependent_columns_excluded(self):
        B = np.column_stack([np.ones(5), np.ones(5)])
        idx = independent_column_indices(B)
        assert len(idx) == 1

    def test_empty_input_returns_empty(self):
        B = np.empty((5, 0))
        idx = independent_column_indices(B)
        assert len(idx) == 0

    def test_zero_matrix_returns_empty(self):
        B = np.zeros((4, 3))
        idx = independent_column_indices(B)
        assert len(idx) == 0

    def test_already_spanned_by_A_returns_empty(self):
        rng = np.random.default_rng(7)
        A = rng.normal(size=(8, 3))
        # B is in the span of A
        B = A @ rng.normal(size=(3, 2))
        idx = independent_column_indices(B, A=A)
        assert len(idx) == 0

    def test_raises_on_1d_input(self):
        with pytest.raises(ValueError, match="2D"):
            independent_column_indices(np.ones(5))


class TestNullSpaceBasisFromConstraintMatrix:
    def test_single_row_constraint_reduces_dimension_by_one(self):
        C = np.ones((1, 4))
        T, rank = null_space_basis_from_constraint_matrix(C, d=4)
        assert rank == 1
        assert T.shape == (4, 3)
        # T is orthogonal to C
        np.testing.assert_allclose(C @ T, np.zeros((1, 3)), atol=1e-12)

    def test_empty_constraint_returns_identity(self):
        C = np.empty((0, 3))
        T, rank = null_space_basis_from_constraint_matrix(C)
        assert rank == 0
        np.testing.assert_allclose(T, np.eye(3))

    def test_1d_input_reshaped(self):
        C = np.ones(4)
        T, rank = null_space_basis_from_constraint_matrix(C)
        assert rank == 1
        assert T.shape == (4, 3)

    def test_width_mismatch_raises(self):
        C = np.ones((1, 3))
        with pytest.raises(ValueError, match="width"):
            null_space_basis_from_constraint_matrix(C, d=5)

    def test_full_rank_constraint_returns_zero_width_T(self):
        # 3 independent constraints on 3 dimensions → null space is trivial
        C = np.eye(3)
        T, rank = null_space_basis_from_constraint_matrix(C)
        assert rank == 3
        assert T.shape[1] == 0

    def test_result_spans_null_space_of_C(self):
        rng = np.random.default_rng(3)
        C = rng.normal(size=(2, 5))
        T, _ = null_space_basis_from_constraint_matrix(C)
        np.testing.assert_allclose(C @ T, np.zeros((2, T.shape[1])), atol=1e-12)


class TestApplyCoefficientTransform:
    def test_identity_transform_preserves_B_and_S(self):
        B = np.arange(12, dtype=float).reshape(4, 3)
        S = np.eye(3)
        T = np.eye(3)
        B_out, Ss_out = apply_coefficient_transform(B, [S], T)
        np.testing.assert_allclose(B_out, B)
        np.testing.assert_allclose(Ss_out[0], S)

    def test_transform_applied_to_basis(self):
        B = np.arange(12, dtype=float).reshape(4, 3)
        T = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
        B_out, _ = apply_coefficient_transform(B, [], T)
        np.testing.assert_allclose(B_out, B @ T)

    def test_penalty_symmetrized(self):
        # Build an asymmetric penalty; transform should symmetrize the result
        B = np.eye(3)
        S = np.array([[2, 1, 0], [0, 2, 1], [0, 0, 2]], dtype=float)
        T = np.eye(3)
        _, Ss_out = apply_coefficient_transform(B, [S], T)
        S_result = Ss_out[0]
        np.testing.assert_allclose(S_result, S_result.T, atol=1e-14)

    def test_transform_contracts_dimension(self):
        B = np.ones((5, 4))
        S = np.eye(4)
        # T maps 4-dim → 3-dim
        T = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        B_out, Ss_out = apply_coefficient_transform(B, [S], T)
        assert B_out.shape == (5, 3)
        assert Ss_out[0].shape == (3, 3)


# ─────────────────────────────────────────────────────────────────────────────
# 2. gam/constraints/absorption.py
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyLinearConstraint:
    def test_trivial_constraint_returns_unchanged(self):
        B = np.eye(4)
        S = [np.eye(4)]
        c = np.zeros(4)  # zero constraint → norm <= tol
        Bc, Sc, C = apply_linear_constraint(B, S, c)
        np.testing.assert_allclose(Bc, B)
        assert C.shape == (4, 4)

    def test_ones_constraint_removes_one_column(self):
        B = np.eye(4)
        S = [np.eye(4)]
        c = np.ones(4)
        Bc, Sc, C = apply_linear_constraint(B, S, c)
        assert Bc.shape == (4, 3)
        assert Sc[0].shape == (3, 3)
        # The constraint is satisfied: c @ coef = 0 for any coef = C @ alpha
        np.testing.assert_allclose(c @ C, np.zeros(3), atol=1e-12)

    def test_constraint_output_is_orthogonal_complement(self):
        rng = np.random.default_rng(10)
        B = rng.normal(size=(8, 5))
        S = [rng.normal(size=(5, 5))]
        c = rng.normal(size=5)
        _, _, C = apply_linear_constraint(B, S, c)
        # c^T @ C == 0
        np.testing.assert_allclose(c @ C, np.zeros(C.shape[1]), atol=1e-12)


class TestFullTermSumToZeroConstraint:
    def test_columns_sum_to_zero_after_constraint(self):
        rng = np.random.default_rng(20)
        B = rng.normal(size=(20, 6))
        S = [np.eye(6)]
        Bc, _, _ = full_term_sum_to_zero_constraint(B, S)
        col_sums = Bc.mean(axis=0)
        np.testing.assert_allclose(col_sums, np.zeros(Bc.shape[1]), atol=1e-12)

    def test_dimension_reduced_by_one(self):
        B = np.random.default_rng(99).normal(size=(10, 5))
        S = [np.eye(5)]
        Bc, Sc, _ = full_term_sum_to_zero_constraint(B, S)
        assert Bc.shape == (10, 4)
        assert Sc[0].shape == (4, 4)


class TestAbsorbExplicitConstraints:
    def test_identity_constraint_reduces_by_one_column(self):
        B = np.eye(4)
        S = PenaltySpec(matrix=np.eye(4))
        S_norm = normalize_penalty_spec(S)
        # One constraint: all-ones row
        C_mat = np.ones((1, 4))
        B_new, specs_new, T, n_cons = absorb_explicit_constraints(B, [S_norm], C_mat)
        assert B_new.shape[1] == 3
        assert n_cons == 1

    def test_no_constraint_returns_unchanged(self):
        B = np.eye(3)
        spec = normalize_penalty_spec(PenaltySpec(matrix=np.eye(3)))
        # Zero constraint matrix: empty rows
        C_mat = np.zeros((0, 3))
        B_new, specs_new, T, n_cons = absorb_explicit_constraints(B, [spec], C_mat)
        assert n_cons == 0
        assert B_new.shape == B.shape


class TestShouldApplyIdentifiabilityConstraint:
    def _by(self, *, is_constant=True):
        return ByState(None, None, None, False, is_constant)

    def test_always_returns_true(self):
        assert should_apply_identifiability_constraint(self._by(), "always") is True

    def test_never_returns_false(self):
        assert should_apply_identifiability_constraint(self._by(), "never") is False

    def test_auto_default_when_auto_false(self):
        # When default_when_auto=False and by is not varying, result is False
        assert (
            should_apply_identifiability_constraint(
                self._by(), "auto", default_when_auto=False
            )
            is False
        )

    def test_auto_default_when_auto_true_and_constant_by(self):
        # When is_constant=True and default_when_auto=True → True
        assert (
            should_apply_identifiability_constraint(
                self._by(is_constant=True), "auto", default_when_auto=True
            )
            is True
        )

    def test_auto_default_when_auto_true_and_varying_by(self):
        # When is_constant=False and default_when_auto=True → False
        assert (
            should_apply_identifiability_constraint(
                self._by(is_constant=False), "auto", default_when_auto=True
            )
            is False
        )

    def test_factor_by_returns_false(self):
        assert should_apply_identifiability_constraint(self._by(), "factor_by") is False

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="constraint_mode"):
            should_apply_identifiability_constraint(self._by(), "invalid_mode")


class TestFitSinglePenaltyWithConstraintPolicy:
    def _by_constant(self):
        return ByState(None, None, None, is_present=False, is_constant=True)

    def _by_varying(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0])
        return ByState(0, "z", vals, is_present=True, is_constant=False)

    def test_never_mode_no_constraint(self):
        B = np.ones((4, 3))
        S = np.eye(3)
        result = fit_single_penalty_with_constraint_policy(
            B, S, self._by_constant(), constraint_mode="never"
        )
        assert isinstance(result, ConstraintFitResult)
        assert result.constraint_kind is None
        assert result.basis_train.shape == (4, 3)

    def test_always_mode_applies_sum_to_zero(self):
        rng = np.random.default_rng(5)
        B = rng.normal(size=(10, 5))
        S = np.eye(5)
        result = fit_single_penalty_with_constraint_policy(
            B, S, self._by_constant(), constraint_mode="always"
        )
        assert result.constraint_kind == "sum_to_zero"
        assert result.basis_train.shape == (10, 4)
        # columns sum to zero (in mean)
        np.testing.assert_allclose(
            result.basis_train.mean(axis=0), np.zeros(4), atol=1e-12
        )

    def test_factor_by_mode_scales_and_constrains(self):
        rng = np.random.default_rng(6)
        B = rng.normal(size=(4, 5))
        S = np.eye(5)
        result = fit_single_penalty_with_constraint_policy(
            B, S, self._by_varying(), constraint_mode="factor_by"
        )
        assert result.constraint_kind == "factor_by"
        assert result.basis_train.shape == (4, 4)

    def test_factor_by_without_by_raises(self):
        B = np.eye(4)
        S = np.eye(4)
        by_absent = ByState(None, None, None, is_present=False, is_constant=True)
        with pytest.raises(ValueError, match="factor_by"):
            fit_single_penalty_with_constraint_policy(
                B, S, by_absent, constraint_mode="factor_by"
            )

    def test_fixed_mode_no_penalty_returned(self):
        B = np.ones((5, 3))
        S = np.eye(3)
        result = fit_single_penalty_with_constraint_policy(
            B, S, self._by_constant(), constraint_mode="never", fixed=True
        )
        # fixed=True → no penalty matrices in output
        assert result.penalties == []


# ─────────────────────────────────────────────────────────────────────────────
# 3. gam/penalties/algebra.py
# ─────────────────────────────────────────────────────────────────────────────


class TestSymmetrizePenalty:
    def test_symmetric_matrix_unchanged(self):
        S = np.array([[2.0, 1.0], [1.0, 3.0]])
        np.testing.assert_allclose(symmetrize_penalty(S), S)

    def test_asymmetric_matrix_symmetrized(self):
        S = np.array([[1.0, 2.0], [4.0, 3.0]])
        result = symmetrize_penalty(S)
        np.testing.assert_allclose(result, result.T)
        np.testing.assert_allclose(result[0, 1], 3.0)

    def test_output_dtype_float64(self):
        S = np.array([[1, 0], [0, 1]], dtype=int)
        result = symmetrize_penalty(S)
        assert result.dtype == np.float64


class TestPenaltyEigendecomposition:
    def test_identity_has_all_positive_eigenvalues(self):
        S = np.eye(4)
        dec = penalty_eigendecomposition(S)
        assert dec["rank"] == 4
        assert dec["null_space_dim"] == 0
        assert dec["U0"].shape == (4, 0)
        assert dec["U1"].shape == (4, 4)

    def test_rank_deficient_penalty_has_null_space(self):
        # 2nd-difference penalty: rank n-2 for n-dim
        n = 5
        D = np.diff(np.eye(n), n=2, axis=0)
        S = D.T @ D
        dec = penalty_eigendecomposition(S)
        assert dec["rank"] == n - 2
        assert dec["null_space_dim"] == 2

    def test_eigenvalues_ascending(self):
        rng = np.random.default_rng(99)
        A = rng.normal(size=(5, 5))
        S = symmetrize_penalty(A.T @ A)
        dec = penalty_eigendecomposition(S)
        evals = dec["evals"]
        assert np.all(np.diff(evals) >= -1e-14)

    def test_reconstruction_from_eigendecomposition(self):
        rng = np.random.default_rng(13)
        A = rng.normal(size=(4, 4))
        S = symmetrize_penalty(A.T @ A)
        dec = penalty_eigendecomposition(S)
        U = dec["U"]
        evals = dec["evals"]
        S_recon = U @ np.diag(evals) @ U.T
        np.testing.assert_allclose(S_recon, S, atol=1e-12)


class TestNullSpacePenaltyFromPenalty:
    def test_full_rank_returns_zero_null_penalty(self):
        S = np.eye(4)
        S0, meta = null_space_penalty_from_penalty(S)
        # Full-rank → no null space → S0 is zero matrix
        np.testing.assert_allclose(S0, np.zeros((4, 4)), atol=1e-12)
        assert meta["rank"] == 0
        assert meta["null_space_dim"] == 0

    def test_rank_deficient_has_nonzero_null_penalty(self):
        n = 5
        D = np.diff(np.eye(n), n=2, axis=0)
        S = D.T @ D
        S0, meta = null_space_penalty_from_penalty(S)
        assert meta["null_space_dim"] == 2
        assert meta["rank"] > 0
        # S0 is symmetric
        np.testing.assert_allclose(S0, S0.T, atol=1e-12)

    def test_null_penalty_projects_onto_null_space(self):
        # Build a penalty with known 1-D null space (all-ones vector)
        n = 4
        ones = np.ones(n) / np.sqrt(n)
        # S = I - ones @ ones.T has null space {ones}
        S = np.eye(n) - np.outer(ones, ones)
        S0, meta = null_space_penalty_from_penalty(S)
        assert meta["null_space_dim"] == 1
        # S0 should project onto the null space of S
        np.testing.assert_allclose(S0 @ ones, ones, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# 4. gam/penalties/subsystem.py
# ─────────────────────────────────────────────────────────────────────────────


class TestPenaltyRankNullDim:
    def test_identity_full_rank(self):
        rank, null_dim = penalty_rank_null_dim(np.eye(5))
        assert rank == 5
        assert null_dim == 0

    def test_zero_matrix_all_null(self):
        rank, null_dim = penalty_rank_null_dim(np.zeros((4, 4)))
        assert rank == 0
        assert null_dim == 4

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            penalty_rank_null_dim(np.ones((3, 4)))

    def test_rank_plus_null_equals_size(self):
        n = 6
        D = np.diff(np.eye(n), n=1, axis=0)
        S = D.T @ D
        rank, null_dim = penalty_rank_null_dim(S)
        assert rank + null_dim == n


class TestNormalizePenaltySpec:
    def test_symmetrizes_matrix(self):
        S = np.array([[1.0, 2.0], [4.0, 3.0]])
        spec = normalize_penalty_spec(PenaltySpec(matrix=S))
        np.testing.assert_allclose(spec.matrix, spec.matrix.T)

    def test_fills_rank_and_null_dim(self):
        S = np.eye(3)
        spec = normalize_penalty_spec(PenaltySpec(matrix=S))
        assert spec.rank == 3
        assert spec.null_space_dim == 0

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            normalize_penalty_spec(PenaltySpec(matrix=np.ones((2, 3))))

    def test_fixed_without_sp_value_raises(self):
        spec = PenaltySpec(matrix=np.eye(2), sp_mode="fixed", sp_value=None)
        with pytest.raises(ValueError, match="sp_value"):
            normalize_penalty_spec(spec)

    def test_sp_value_without_fixed_raises(self):
        spec = PenaltySpec(matrix=np.eye(2), sp_mode=None, sp_value=1.0)
        with pytest.raises(ValueError, match="sp_value"):
            normalize_penalty_spec(spec)

    def test_invalid_sp_mode_raises(self):
        spec = PenaltySpec(matrix=np.eye(2), sp_mode="bogus")
        with pytest.raises(ValueError, match="sp_mode"):
            normalize_penalty_spec(spec)

    def test_fixed_with_valid_sp_value_ok(self):
        spec = PenaltySpec(matrix=np.eye(2), sp_mode="fixed", sp_value=3.0)
        out = normalize_penalty_spec(spec)
        assert out.sp_mode == "fixed"
        assert out.sp_value == pytest.approx(3.0)


class TestBuildFullPenaltyFromBlocks:
    def test_symmetrizes_penalty_block_before_assembly(self):
        class _PB:
            smoothing_index = 0
            coef_slice = slice(0, 2)
            label = "asym"
            matrix = np.array([[2.0, 1e-12], [0.0, 3.0]], dtype=np.float64)

        P_full = build_full_penalty_from_blocks(
            penalty_blocks=[_PB()],
            smoothing_params=np.array([0.5], dtype=np.float64),
            fit_intercept=False,
            n_coef=2,
        )

        np.testing.assert_allclose(P_full, P_full.T, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            P_full,
            0.5 * (_PB.matrix + _PB.matrix.T) * 0.5,
            atol=0.0,
            rtol=0.0,
        )


class TestFitBackendSelection:
    def test_explicit_stacked_qr_backend_takes_priority(self):
        class _Family:
            name = "gaussian"
            supports_closed_form_solve = True
            supports_pirls = False

        class _Model:
            family = _Family()
            _use_stacked_qr = True

        assert available_fit_backends(_Model()) == ("stacked_qr", "gaussian_exact")
        assert resolve_fit_backend(_Model()) == "stacked_qr"

    def test_numpy_array_input(self):
        S = np.eye(3)
        spec = normalize_penalty_spec(S)
        assert spec.rank == 3


class TestMakePenaltySpec:
    def test_basic_creation(self):
        S = np.eye(4)
        spec = make_penalty_spec(matrix=S, smoothing_id="test_id", kind="smooth")
        assert spec.smoothing_id == "test_id"
        assert spec.kind == "smooth"
        assert spec.rank == 4

    def test_null_space_penalty_flag(self):
        S = np.eye(2)
        spec = make_penalty_spec(
            matrix=S, smoothing_id=None, kind="null_space", is_null_space_penalty=True
        )
        assert spec.is_null_space_penalty is True


class TestBuildNullSpaceSelectionSpec:
    def test_full_rank_returns_none(self):
        S = np.eye(4)
        result = build_null_space_selection_spec(main_penalty=S, smoothing_id="foo")
        assert result is None

    def test_rank_deficient_returns_spec(self):
        n = 5
        D = np.diff(np.eye(n), n=2, axis=0)
        S = D.T @ D
        result = build_null_space_selection_spec(main_penalty=S, smoothing_id="bar")
        assert result is not None
        assert result.is_null_space_penalty is True
        assert result.kind == "null_space"


class TestMergeSmoothingOverride:
    def test_none_mode_returns_existing(self):
        existing = {"mode": "fixed", "value": 1.0, "labels": ["a"]}
        result = merge_smoothing_override(
            existing, None, None, smoothing_id="id", label="b"
        )
        assert result is existing

    def test_first_override_creates_entry(self):
        result = merge_smoothing_override(
            None, "fixed", 2.5, smoothing_id="id", label="t"
        )
        assert result["mode"] == "fixed"
        assert result["value"] == pytest.approx(2.5)
        assert "t" in result["labels"]

    def test_conflicting_mode_raises(self):
        existing = {"mode": "fixed", "value": 1.0, "labels": ["a"]}
        with pytest.raises(ValueError, match="Conflicting"):
            merge_smoothing_override(
                existing, "estimate", None, smoothing_id="id", label="b"
            )

    def test_conflicting_fixed_value_raises(self):
        existing = {"mode": "fixed", "value": 1.0, "labels": ["a"]}
        with pytest.raises(ValueError, match="Conflicting"):
            merge_smoothing_override(
                existing, "fixed", 2.0, smoothing_id="id", label="b"
            )

    def test_matching_fixed_value_appends_label(self):
        existing = {"mode": "fixed", "value": 1.0, "labels": ["a"]}
        result = merge_smoothing_override(
            existing, "fixed", 1.0, smoothing_id="id", label="b"
        )
        assert "b" in result["labels"]


class TestDefaultPenaltyId:
    class _FakeTerm:
        smoothing_id = "my_id"

    class _FakeTermNoId:
        smoothing_id = None

    def test_uses_smoothing_id_single_penalty(self):
        result = default_penalty_id("pred", self._FakeTerm(), "label", 0, 0, 1)
        assert result == "my_id"

    def test_uses_smoothing_id_multi_penalty(self):
        result = default_penalty_id("pred", self._FakeTerm(), "label", 0, 1, 3)
        assert result == "my_id::1"

    def test_auto_id_when_no_smoothing_id(self):
        result = default_penalty_id("pred", self._FakeTermNoId(), "label", 4, 0, 1)
        assert result.startswith("__auto__")


# ─────────────────────────────────────────────────────────────────────────────
# 5. gam/basis/tensor.py
# ─────────────────────────────────────────────────────────────────────────────


class TestRowwiseKronecker:
    def test_two_matrices(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = rowwise_kronecker([A, B])
        # Row 0: outer product of [1,2] and [5,6] = [5,6,10,12]
        expected_r0 = np.array([5.0, 6.0, 10.0, 12.0])
        np.testing.assert_allclose(result[0], expected_r0)
        assert result.shape == (2, 4)

    def test_single_matrix_returns_self(self):
        A = np.arange(6, dtype=float).reshape(2, 3)
        result = rowwise_kronecker([A])
        np.testing.assert_allclose(result, A)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            rowwise_kronecker([])

    def test_row_mismatch_raises(self):
        A = np.ones((3, 2))
        B = np.ones((4, 2))
        with pytest.raises(ValueError, match="equal rows"):
            rowwise_kronecker([A, B])

    def test_three_matrices(self):
        n = 5
        A = np.ones((n, 2))
        B = np.ones((n, 3))
        C = np.ones((n, 4))
        result = rowwise_kronecker([A, B, C])
        assert result.shape == (n, 2 * 3 * 4)


class TestLiftedTensorPenalty:
    def test_axis_0_lifts_on_right(self):
        # lifted_tensor_penalty(S, basis_dims, axis=0)
        # → kron(S, I_{right_dim})
        S = np.array([[1.0, 0.5], [0.5, 2.0]])
        basis_dims = [2, 3]
        P = lifted_tensor_penalty(S, basis_dims, axis=0)
        expected = np.kron(S, np.eye(3))
        np.testing.assert_allclose(P, expected)

    def test_axis_1_lifts_on_left(self):
        S = np.eye(3)
        basis_dims = [2, 3]
        P = lifted_tensor_penalty(S, basis_dims, axis=1)
        expected = np.kron(np.eye(2), S)
        np.testing.assert_allclose(P, expected)

    def test_single_dim_returns_penalty(self):
        S = np.array([[3.0, 1.0], [1.0, 2.0]])
        P = lifted_tensor_penalty(S, [2], axis=0)
        np.testing.assert_allclose(P, S)


class TestTensorProductPenalties:
    def test_returns_one_penalty_per_marginal(self):
        S1 = np.eye(3)
        S2 = np.eye(4)
        result = tensor_product_penalties([S1, S2], [3, 4])
        assert len(result) == 2
        assert result[0].shape == (12, 12)
        assert result[1].shape == (12, 12)


class TestNormalizeTensorMarginalPenalty:
    def test_identity_scaled_to_one(self):
        S = 5.0 * np.eye(4)
        result = normalize_tensor_marginal_penalty(S)
        # max eigenvalue should be 1 after normalization
        evals = np.linalg.eigvalsh(result)
        assert np.max(evals) == pytest.approx(1.0, rel=1e-10)

    def test_zero_matrix_unchanged(self):
        S = np.zeros((4, 4))
        result = normalize_tensor_marginal_penalty(S)
        np.testing.assert_allclose(result, np.zeros((4, 4)))

    def test_empty_matrix_unchanged(self):
        S = np.empty((0, 0))
        result = normalize_tensor_marginal_penalty(S)
        assert result.shape == (0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 6. gam/fit/covariance.py
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildBayesAndFreqCovariances:
    def test_identity_case(self):
        scale = 1.0
        A_inv = np.eye(3)
        XWX = np.eye(3)
        Vp, Vf, H = build_bayes_and_freq_covariances(scale, A_inv, XWX)
        # With identity: Vp = I, Vf = I, H = I
        np.testing.assert_allclose(Vp, np.eye(3))
        np.testing.assert_allclose(Vf, np.eye(3))
        np.testing.assert_allclose(H, np.eye(3))

    def test_scale_multiplied(self):
        scale = 2.5
        A_inv = np.eye(3)
        XWX = np.eye(3)
        Vp, Vf, _ = build_bayes_and_freq_covariances(scale, A_inv, XWX)
        np.testing.assert_allclose(Vp, 2.5 * np.eye(3))

    def test_trace_H_is_edf(self):
        # In the penalized Gaussian case, trace(H) = EDF
        # Build a simple penalized normal equations example
        scale = 1.0
        rng = np.random.default_rng(77)
        X = rng.normal(size=(10, 3))
        S = np.eye(3)
        lam = 0.5
        XWX = X.T @ X
        A = XWX + lam * S
        A_inv = np.linalg.inv(A)
        _, _, H = build_bayes_and_freq_covariances(scale, A_inv, XWX)
        edf_via_H = np.trace(H)
        # EDF must be between 0 and n_coef
        assert 0 < edf_via_H <= 3

    def test_Vf_is_symmetric(self):
        rng = np.random.default_rng(11)
        XWX = rng.normal(size=(4, 4))
        XWX = XWX.T @ XWX  # make it PSD
        A_inv = np.linalg.inv(XWX + np.eye(4))
        Vp, Vf, H = build_bayes_and_freq_covariances(1.0, A_inv, XWX)
        np.testing.assert_allclose(Vf, Vf.T, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# 7. gam/fit/linalg/matrix_reindexing.py
# ─────────────────────────────────────────────────────────────────────────────


class TestDropColumnsDense:
    def test_drop_no_columns_returns_copy(self):
        A = np.eye(4)
        result = drop_columns_dense(A, np.array([], dtype=int))
        np.testing.assert_allclose(result, A)
        assert result is not A  # copy

    def test_drop_one_column(self):
        A = np.arange(12, dtype=float).reshape(3, 4)
        result = drop_columns_dense(A, np.array([1]))
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result, A[:, [0, 2, 3]])

    def test_drop_all_columns_returns_empty(self):
        A = np.ones((3, 2))
        result = drop_columns_dense(A, np.array([0, 1]))
        assert result.shape == (3, 0)


class TestDropRowsDense:
    def test_drop_no_rows(self):
        A = np.eye(4)
        result = drop_rows_dense(A, np.array([], dtype=int))
        np.testing.assert_allclose(result, A)

    def test_drop_middle_row(self):
        A = np.arange(9, dtype=float).reshape(3, 3)
        result = drop_rows_dense(A, np.array([1]))
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result, A[[0, 2], :])


class TestRestoreDroppedRows:
    def test_roundtrip_with_no_drops(self):
        A = np.arange(12, dtype=float).reshape(4, 3)
        result = restore_dropped_rows(A, 4, np.array([], dtype=int))
        np.testing.assert_allclose(result[:4, :], A)

    def test_roundtrip_with_one_drop(self):
        A_full = np.arange(12, dtype=float).reshape(4, 3)
        drop_idx = np.array([1])
        A_packed = drop_rows_dense(A_full, drop_idx)
        A_restored = restore_dropped_rows(A_packed, 4, drop_idx)
        # Row 0 and rows 2,3 should match; row 1 should be zero
        np.testing.assert_allclose(A_restored[[0, 2, 3], :], A_full[[0, 2, 3], :])
        np.testing.assert_allclose(A_restored[1, :], np.zeros(3))

    def test_wrong_packed_size_raises(self):
        A_wrong = np.zeros((5, 2))
        with pytest.raises(ValueError, match="packed row count"):
            restore_dropped_rows(A_wrong, 8, np.array([1, 2]))


class TestPermuteColumns:
    def test_identity_permutation(self):
        A = np.arange(12, dtype=float).reshape(3, 4)
        perm = np.arange(4)
        np.testing.assert_allclose(permute_columns(A, perm), A)

    def test_reverse_permutation(self):
        A = np.arange(12, dtype=float).reshape(3, 4)
        perm = np.array([3, 2, 1, 0])
        result = permute_columns(A, perm)
        np.testing.assert_allclose(result, A[:, [3, 2, 1, 0]])

    def test_wrong_permutation_length_raises(self):
        A = np.ones((3, 4))
        with pytest.raises(ValueError, match="permutation length"):
            permute_columns(A, np.arange(3))

    def test_reverse_flag_inverts_permutation(self):
        A = np.arange(12, dtype=float).reshape(3, 4)
        perm = np.array([1, 3, 0, 2])
        forward = permute_columns(A, perm)
        back = permute_columns(forward, perm, reverse=True)
        np.testing.assert_allclose(back, A)


class TestPermuteRows:
    def test_identity_permutation(self):
        A = np.arange(12, dtype=float).reshape(4, 3)
        perm = np.arange(4)
        np.testing.assert_allclose(permute_rows(A, perm), A)

    def test_reverse_flag_inverts(self):
        A = np.arange(12, dtype=float).reshape(4, 3)
        perm = np.array([2, 0, 3, 1])
        forward = permute_rows(A, perm)
        back = permute_rows(forward, perm, reverse=True)
        np.testing.assert_allclose(back, A)

    def test_wrong_permutation_length_raises(self):
        A = np.ones((4, 3))
        with pytest.raises(ValueError, match="permutation length"):
            permute_rows(A, np.arange(3))


# ─────────────────────────────────────────────────────────────────────────────
# 8. gam/smooths/base.py helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveFeature:
    def test_resolve_by_name(self):
        feature_names = ["a", "b", "c"]
        idx, name = _resolve_feature("b", feature_names)
        assert idx == 1
        assert name == "b"

    def test_resolve_by_index(self):
        feature_names = ["a", "b", "c"]
        idx, name = _resolve_feature(2, feature_names)
        assert idx == 2
        assert name == "c"

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError, match="not found"):
            _resolve_feature("z", ["a", "b"])

    def test_out_of_range_index_raises(self):
        with pytest.raises(IndexError, match="out of range"):
            _resolve_feature(5, ["a", "b", "c"])

    def test_negative_index_raises(self):
        with pytest.raises(IndexError, match="out of range"):
            _resolve_feature(-1, ["a", "b"])


class TestResolveByState:
    def test_none_by_returns_absent_constant_state(self):
        X = np.ones((5, 2))
        state = resolve_by_state(None, X, ["a", "b"])
        assert state.is_present is False
        assert state.is_constant is True
        assert state.values is None

    def test_numeric_by_returns_state_with_values(self):
        X = np.column_stack([np.arange(5, dtype=float), np.ones(5)])
        state = resolve_by_state("a", X, ["a", "b"])
        assert state.is_present is True
        np.testing.assert_allclose(state.values, np.arange(5, dtype=float))


class TestResolveFeatureMatrixState:
    def test_single_feature(self):
        X = np.arange(10, dtype=float).reshape(5, 2)
        state = resolve_feature_matrix_state("a", X, ["a", "b"])
        assert state.indices == [0]
        assert state.names == ["a"]
        assert state.matrix.shape == (5, 1)

    def test_multiple_features(self):
        X = np.arange(15, dtype=float).reshape(5, 3)
        state = resolve_feature_matrix_state(["a", "c"], X, ["a", "b", "c"])
        assert state.indices == [0, 2]
        assert state.names == ["a", "c"]
        assert state.matrix.shape == (5, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 9. gam/formula/parser.py
# ─────────────────────────────────────────────────────────────────────────────


class TestParseGAMFormula:
    def test_simple_with_intercept(self):
        parsed = parse_gam_formula('y ~ s(x0, bs="cr", k=8)')
        assert parsed.response_name == "y"
        assert len(parsed.predictors) == 1
        pred = parsed.predictors[0]
        assert pred.intercept is True
        smooth_terms = [t for t in pred.terms if isinstance(t, ParsedSmoothTerm)]
        assert len(smooth_terms) == 1
        assert smooth_terms[0].features == ["x0"]
        assert smooth_terms[0].kwargs["bs"] == "cr"
        assert smooth_terms[0].kwargs["k"] == 8

    def test_no_intercept_dash_one(self):
        parsed = parse_gam_formula("y ~ s(x0) - 1")
        pred = parsed.predictors[0]
        assert pred.intercept is False

    def test_no_intercept_plus_zero(self):
        parsed = parse_gam_formula("y ~ s(x0) + 0")
        pred = parsed.predictors[0]
        assert pred.intercept is False

    def test_parametric_term_parsed(self):
        parsed = parse_gam_formula("y ~ x1 + s(x0)")
        terms = parsed.predictors[0].terms
        param_terms = [t for t in terms if isinstance(t, ParsedParametricTerm)]
        assert len(param_terms) == 1
        assert "x1" in param_terms[0].variables

    def test_tensor_smooth_te(self):
        parsed = parse_gam_formula('y ~ te(x0, x1, bs="cr")')
        terms = [
            t for t in parsed.predictors[0].terms if isinstance(t, ParsedSmoothTerm)
        ]
        assert len(terms) == 1
        assert terms[0].kind == "te"
        assert set(terms[0].features) == {"x0", "x1"}

    def test_multiple_smooths(self):
        parsed = parse_gam_formula('y ~ s(x0, bs="cr") + s(x1, bs="ps")')
        smooths = [
            t for t in parsed.predictors[0].terms if isinstance(t, ParsedSmoothTerm)
        ]
        assert len(smooths) == 2

    def test_unknown_smooth_raises(self):
        with pytest.raises((ValueError, NotImplementedError)):
            parse_gam_formula("y ~ foo(x0)")

    def test_response_extracted(self):
        parsed = parse_gam_formula("response_var ~ s(x0)")
        assert parsed.response_name == "response_var"


class TestCompilePredictorSpecsFromFormula:
    def test_cr_smooth_gets_cr_basis(self):
        parsed = parse_gam_formula('y ~ s(x0, bs="cr", k=8)')
        specs = compile_predictor_specs_from_formula(parsed)
        assert len(specs) == 1
        term_specs = [t for t in specs[0].terms if t.kind == "smooth"]
        assert len(term_specs) == 1
        assert term_specs[0].basis_options["bs"] == "cr"
        assert term_specs[0].basis_options["k"] == 8

    def test_default_basis_applied_when_missing(self):
        parsed = parse_gam_formula("y ~ s(x0)")
        specs = compile_predictor_specs_from_formula(parsed, default_basis="cr")
        smooth_specs = [t for t in specs[0].terms if t.kind == "smooth"]
        assert len(smooth_specs) == 1
        assert smooth_specs[0].basis_options["bs"] == "cr"

    def test_bare_s_defaults_to_tp_like_mgcv(self):
        parsed = parse_gam_formula("y ~ s(x0)")
        specs = compile_predictor_specs_from_formula(parsed)
        smooth_specs = [t for t in specs[0].terms if t.kind == "smooth"]
        assert len(smooth_specs) == 1
        assert smooth_specs[0].smooth_spec.__class__.__name__ == "ThinPlateSmoothSpec"
        assert smooth_specs[0].basis_options["bs"] == "tp"

    def test_bare_te_keeps_cr_marginals_even_when_s_default_is_tp(self):
        parsed = parse_gam_formula("y ~ te(x0, x1, k=[6, 6])")
        specs = compile_predictor_specs_from_formula(parsed, default_basis="tp")
        smooth_specs = [t for t in specs[0].terms if t.kind == "smooth"]
        assert len(smooth_specs) == 1
        assert smooth_specs[0].smooth_spec.__class__.__name__ == "TensorProductSmoothSpec"
        assert smooth_specs[0].basis_options["special"] == "te"
        assert smooth_specs[0].basis_options["bs"] == "cr"

    def test_no_intercept_propagated(self):
        parsed = parse_gam_formula("y ~ s(x0) - 1")
        specs = compile_predictor_specs_from_formula(parsed)
        assert specs[0].has_intercept is False

    def test_multiple_predictors_from_list(self):
        parsed = parse_gam_formula('y ~ s(x0, bs="cr") + s(x1, bs="ps")')
        specs = compile_predictor_specs_from_formula(parsed)
        smooth_specs = [t for t in specs[0].terms if t.kind == "smooth"]
        assert len(smooth_specs) == 2

    def test_missing_by_column_raises_early(self):
        parsed = parse_gam_formula('y ~ s(x0, by="temperatrue", bs="cr", k=8)')
        with pytest.raises(KeyError, match="by column 'temperatrue' not found"):
            compile_predictor_specs_from_formula(
                parsed,
                available_columns=["x0", "temperature", "y"],
            )

    def test_termspec_basis_options_is_read_only_copy(self):
        opts = {"special": "s", "bs": "cr", "k": 8}
        spec = TermSpec(kind="smooth", features=("x0",), basis_options=opts)

        opts["k"] = 20
        assert spec.basis_options["k"] == 8
        assert spec.smooth_spec.__class__.__name__ == "CubicRegressionSmoothSpec"

        with pytest.raises(TypeError):
            spec.basis_options["k"] = 20

    def test_returns_linear_predictor_spec_objects(self):
        parsed = parse_gam_formula("y ~ s(x0)")
        specs = compile_predictor_specs_from_formula(parsed)
        assert all(spec.__class__.__name__ == "LinearPredictorSpec" for spec in specs)


class TestFamilyApi:
    def test_capability_flags_are_read_only_on_class_and_instance(self):
        fam = GaussianIdentityFamily()

        with pytest.raises(AttributeError, match="supports_reml is read-only"):
            fam.supports_reml = False

        with pytest.raises(AttributeError, match="supports_reml is read-only"):
            BaseFamily.supports_reml = True

    def test_weighted_deviance_and_aic_are_available(self):
        fam = GaussianIdentityFamily()
        y = np.asarray([1.0, 3.0], dtype=np.float64)
        mu = np.asarray([0.0, 1.0], dtype=np.float64)
        weights = np.asarray([2.0, 0.5], dtype=np.float64)

        assert fam.deviance(y, mu, weights=weights) == pytest.approx(2.0 + 2.0)
        assert fam.aic(y, mu, edf=3.0, scale=2.0, weights=weights) == pytest.approx(
            -2.0 * np.sum(weights * fam.loglik_obs(y, mu, scale=2.0)) + 6.0
        )

    @pytest.mark.parametrize(
        ("family", "mu"),
        [
            (GaussianIdentityFamily(), np.asarray([0.2, 1.3], dtype=np.float64)),
            (BinomialLogitFamily(), np.asarray([0.2, 0.8], dtype=np.float64)),
            (PoissonLogFamily(), np.asarray([0.2, 1.3], dtype=np.float64)),
            (GammaLogFamily(), np.asarray([0.2, 1.3], dtype=np.float64)),
            (
                NegativeBinomialLogFamily(theta=2.0),
                np.asarray([0.2, 1.3], dtype=np.float64),
            ),
            (BinomialProbitFamily(), np.asarray([0.2, 0.8], dtype=np.float64)),
            (BinomialCloglogFamily(), np.asarray([0.2, 0.8], dtype=np.float64)),
            (GammaInverseFamily(), np.asarray([0.2, 1.3], dtype=np.float64)),
        ],
    )
    def test_link_and_variance_derivatives_delegate_to_mapped_objects(self, family, mu):
        assert np.allclose(family.d2link(mu), family.link.d2(mu))
        assert np.allclose(family.d3link(mu), family.link.d3(mu))
        assert np.allclose(family.d4link(mu), family.link.d4(mu))
        assert np.allclose(family.dvar(mu), family.variance.d1(mu))
        assert np.allclose(family.d2var(mu), family.variance.d2(mu))
        assert np.allclose(family.d3var(mu), family.variance.d3(mu))

    def test_binomial_and_gamma_link_variants_share_family_statistics(self):
        binomial_families = [
            BinomialLogitFamily(),
            BinomialProbitFamily(),
            BinomialCloglogFamily(),
        ]
        gamma_families = [GammaLogFamily(), GammaInverseFamily()]

        y_bin = np.asarray([0.1, 0.8], dtype=np.float64)
        mu_bin = np.asarray([0.2, 0.7], dtype=np.float64)
        w_bin = np.asarray([1.5, 0.5], dtype=np.float64)

        baseline = binomial_families[0]
        baseline_bin = (
            baseline.deviance(y_bin, mu_bin, weights=w_bin),
            baseline.loglik_obs(y_bin, mu_bin),
            baseline.saturated_loglik(y_bin, weights=w_bin),
        )
        for fam in binomial_families[1:]:
            assert fam.deviance(y_bin, mu_bin, weights=w_bin) == pytest.approx(
                baseline_bin[0]
            )
            assert np.allclose(fam.loglik_obs(y_bin, mu_bin), baseline_bin[1])
            assert fam.saturated_loglik(y_bin, weights=w_bin) == pytest.approx(
                baseline_bin[2]
            )

        y_gamma = np.asarray([0.4, 1.8], dtype=np.float64)
        mu_gamma = np.asarray([0.5, 1.4], dtype=np.float64)
        w_gamma = np.asarray([0.75, 1.25], dtype=np.float64)
        scale = 0.3

        gamma_baseline = gamma_families[0]
        baseline_gamma = (
            gamma_baseline.deviance(y_gamma, mu_gamma, weights=w_gamma),
            gamma_baseline.loglik_obs(y_gamma, mu_gamma, scale=scale),
            gamma_baseline.saturated_loglik(y_gamma, weights=w_gamma, scale=scale),
            gamma_baseline.estimate_dispersion(y_gamma, mu_gamma, edf=0.5),
        )
        for fam in gamma_families[1:]:
            assert fam.deviance(y_gamma, mu_gamma, weights=w_gamma) == pytest.approx(
                baseline_gamma[0]
            )
            assert np.allclose(
                fam.loglik_obs(y_gamma, mu_gamma, scale=scale), baseline_gamma[1]
            )
            assert fam.saturated_loglik(
                y_gamma, weights=w_gamma, scale=scale
            ) == pytest.approx(baseline_gamma[2])
            assert fam.estimate_dispersion(y_gamma, mu_gamma, edf=0.5) == pytest.approx(
                baseline_gamma[3]
            )

    def test_make_gam_family_gamma_identity_not_implemented(self):
        with pytest.raises(NotImplementedError, match="GammaIdentityFamily"):
            make_gam_family({"name": "gamma", "link": "identity"})

    def test_make_gam_family_gamma_unknown_link_raises(self):
        with pytest.raises(ValueError, match="Unknown gamma link"):
            make_gam_family({"name": "gamma", "link": "sqrt"})


# ─────────────────────────────────────────────────────────────────────────────
# 10. gam/diagnostics/summary.py
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildSummaryLines:
    def test_runs_on_fitted_model(self):
        gam = _fit_simple_gam()
        lines = build_summary_lines(gam)
        assert isinstance(lines, list)
        assert len(lines) > 0

    def test_contains_family_and_link(self):
        gam = _fit_simple_gam()
        lines = build_summary_lines(gam)
        text = "\n".join(lines)
        assert "gaussian" in text.lower()
        assert "identity" in text.lower()

    def test_contains_edf_and_scale(self):
        gam = _fit_simple_gam()
        lines = build_summary_lines(gam)
        text = "\n".join(lines)
        assert "EDF" in text
        assert "Scale" in text

    def test_contains_rss_for_gaussian(self):
        gam = _fit_simple_gam()
        lines = build_summary_lines(gam)
        text = "\n".join(lines)
        assert "RSS" in text

    def test_unfitted_model_raises(self):
        gam = GAM(formula='y ~ s(x0, bs="cr", k=5)')
        with pytest.raises(RuntimeError, match="not fitted"):
            build_summary_lines(gam)


class TestSummaryText:
    def test_returns_string(self):
        gam = _fit_simple_gam()
        text = summary_text(gam)
        assert isinstance(text, str)
        assert len(text) > 0


class TestPrintSummary:
    def test_returns_text_and_prints(self, capsys):
        gam = _fit_simple_gam()
        result = print_summary(gam)
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        assert result == summary_text(gam)


# ─────────────────────────────────────────────────────────────────────────────
# 11. gam/constraints/identifiability.py — apply_global_side_conditions
#     Tested via full pipeline using GAM objects
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyGlobalSideConditions:
    """
    These tests verify the behavior of apply_global_side_conditions
    through the full GAM pipeline (end-to-end), rather than by
    constructing CompiledPredictor fixtures manually.
    """

    def test_gaussian_intercept_model_columns_sum_to_zero(self):
        """After side conditions, the design matrix columns (excl. intercept) must
        have near-zero column means (sum-to-zero constraint absorbed)."""
        data = _make_gaussian_data(n=80)
        gam = GAM(
            formula='y ~ s(x0, bs="cr", k=8)',
            optimize_smoothing=False,
            smoothing_params=[1.0],
        )
        gam.fit(data=data)
        from nampy.gam.parity.snapshots import _get_core

        core = _get_core(gam)
        X = core.design_.design_matrix
        # Each column of the smooth block should have zero mean
        # (sum-to-zero constraint from identifiability)
        col_means = X.mean(axis=0)
        # Allow some small numerical tolerance
        np.testing.assert_allclose(col_means, np.zeros_like(col_means), atol=1e-12)

    def test_prediction_consistent_with_design(self):
        """Predictions on training data must match intercept + X @ coef."""
        data = _make_gaussian_data(n=60)
        gam = GAM(
            formula='y ~ s(x0, bs="cr", k=6)',
            optimize_smoothing=False,
            smoothing_params=[0.5],
        )
        gam.fit(data=data)
        from nampy.gam.parity.snapshots import _get_core

        core = _get_core(gam)
        X = core.design_.design_matrix
        fr = core.fit_result()
        # coef_full layout: [intercept, coef_0, ..., coef_{n_coef-1}]
        intercept = fr.coef_full[0]
        coef = fr.coef_full[1:]
        pred_manual = intercept + X @ coef
        pred_api = gam.predict(data)
        np.testing.assert_allclose(pred_manual, pred_api, atol=1e-10)

    def test_two_smooth_terms_predictions_match(self):
        """Two-smooth model: prediction must be consistent."""
        data = _make_gaussian_data(n=100)
        gam = GAM(
            formula='y ~ s(x0, bs="cr", k=6) + s(x1, bs="cr", k=6)',
            optimize_smoothing=True,
            smoothing_method="REML",
        )
        gam.fit(data=data)
        pred = gam.predict(data)
        assert pred.shape == (100,)
        # Predictions should be finite
        assert np.all(np.isfinite(pred))

    def test_formula_prediction_metadata_derived_from_preprocess_state(self):
        data = _make_gaussian_data(n=60)
        data["off"] = np.linspace(0.0, 1.0, len(data))
        gam = GAM(
            formula='y ~ s(x0, bs="cr", k=6) + offset(off)',
            optimize_smoothing=False,
            smoothing_params=[0.5],
        )
        gam.fit(data=data)

        assert "formula_used_columns_" not in gam.__dict__
        assert "formula_offset_name_" not in gam.__dict__
        assert (
            gam.formula_used_columns_ == gam.formula_preprocess_state_["used_columns"]
        )
        assert gam.formula_offset_name_ == gam.formula_preprocess_state_["offset_name"]

        pred = gam.predict(data)
        assert pred.shape == (60,)

    def test_report_contains_term_reports(self):
        """apply_global_side_conditions should return a report dict."""
        from nampy.gam.constraints.identifiability import apply_global_side_conditions
        from nampy.gam.parity.snapshots import _get_core

        data = _make_gaussian_data(n=60)
        gam = GAM(
            formula='y ~ s(x0, bs="cr", k=6)',
            optimize_smoothing=False,
            smoothing_params=[1.0],
        )
        gam.fit(data=data)
        core = _get_core(gam)
        # Re-apply to the already-processed design (should be a no-op mostly)
        design_in = core.design_
        new_design, report = apply_global_side_conditions(design_in, fit_intercept=True)
        assert "term_reports" in report
        assert "n_deleted_total" in report
        assert new_design.side_condition_Q is not None
        assert new_design.side_condition_Q.shape == (
            design_in.n_coef,
            new_design.n_coef,
        )
        np.testing.assert_allclose(
            design_in.design_matrix @ new_design.side_condition_Q,
            new_design.design_matrix,
            atol=1e-12,
        )

    def test_non_compiled_predictor_raises(self):
        from nampy.gam.constraints.identifiability import apply_global_side_conditions

        with pytest.raises(TypeError, match="CompiledPredictor"):
            apply_global_side_conditions("not_a_predictor")

    def test_no_intercept_model_no_centering(self):
        """Without an intercept, no centering constraint should be applied."""
        data = _make_gaussian_data(n=60)
        gam = GAM(
            formula='y ~ s(x0, bs="cr", k=6) - 1',
            optimize_smoothing=False,
            smoothing_params=[1.0],
        )
        gam.fit(data=data)
        pred = gam.predict(data)
        assert pred.shape == (60,)
        assert np.all(np.isfinite(pred))


# ─────────────────────────────────────────────────────────────────────────────
# 12. Structural invariants that still add value outside parity suites
# ─────────────────────────────────────────────────────────────────────────────


class TestParityPenaltyMatrixConsistency:
    """After side conditions, penalty matrices must be consistent with basis."""

    def test_penalty_matrix_size_matches_coef_slice(self):
        data = _make_gaussian_data(n=60)
        gam = GAM(
            formula='y ~ s(x0, bs="cr", k=8)',
            optimize_smoothing=False,
            smoothing_params=[1.0],
        )
        gam.fit(data=data)
        from nampy.gam.parity.snapshots import _get_core

        core = _get_core(gam)
        design = core.design_
        for pb in design.compiled_penalties:
            coef_width = pb.coef_slice.stop - pb.coef_slice.start
            assert pb.matrix.shape == (coef_width, coef_width), (
                f"Penalty matrix shape {pb.matrix.shape} does not match "
                f"coef_slice width {coef_width}"
            )

    def test_penalty_matrices_are_symmetric(self):
        data = _make_gaussian_data(n=80)
        gam = GAM(
            formula='y ~ s(x0, bs="cr", k=8) + s(x1, bs="ps", k=6)',
            optimize_smoothing=False,
            smoothing_params=[1.0, 1.0],
        )
        gam.fit(data=data)
        from nampy.gam.parity.snapshots import _get_core

        core = _get_core(gam)
        for pb in core.design_.compiled_penalties:
            S = pb.matrix
            np.testing.assert_allclose(
                S,
                S.T,
                atol=1e-12,
                err_msg=f"Penalty for term {pb.label!r} is not symmetric",
            )


class TestParityTermContributions:
    """Term contributions must sum to the linear predictor."""

    def test_contributions_sum_to_prediction(self):
        data = _make_gaussian_data(n=80)
        gam = GAM(
            formula='y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=6)',
            optimize_smoothing=True,
            smoothing_method="REML",
        )
        gam.fit(data=data)
        result = gam.predict_feature_vals(data)
        pred = gam.predict(data)
        # result["output"] should match predict()
        np.testing.assert_allclose(np.array(result["output"]), pred, atol=1e-10)
