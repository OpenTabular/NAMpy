import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult

from nampy.gam import GAM
from nampy.gam._model_state import (
    _design_matrix,
    _fit_state,
    _n_coef,
    _penalty_blocks_seq,
)
from nampy.gam.families import BinomialLogitFamily, GaussianIdentityFamily
from nampy.gam.fit.linalg.stacked_qr import (
    _stacked_penalized_ls_nonneg_solution,
    balanced_penalty_template_sqrt_for_rank,
)
from nampy.gam.fit.penalized_system import (
    build_full_design,
    build_full_penalty_from_blocks,
)
from nampy.gam.fit.solvers.irls_core import irls_core
from nampy.gam.smoothing_selection.criteria import dispatch as criteria_dispatch
from nampy.gam.smoothing_selection.criteria.pirls_deriv import (
    criterion_gradient_ml_reml_pirls_exact,
)
from nampy.gam.smoothing_selection.optimize.outer import _optimize_outer_newton
from nampy.gam.smoothing_selection.reparam import (
    SlBlock,
    build_estimate_gam_setup_state,
    build_gam_fit3_reparam_state,
    build_penalty_reparameterized_system,
    can_use_simple_ml_reml_structure,
    dynamic_reparam_design,
    sl_group_indices,
)
from tests.mgcv_parity_utils import _make_gaussian_data, _make_mrf_data


def _attach_compiled_model(
    model,
    *,
    design_matrix,
    compiled_terms,
    compiled_penalties,
    metadata=None,
    n_coef=None,
    n_smoothing_params=None,
):
    if n_coef is None:
        n_coef = int(np.asarray(design_matrix, dtype=np.float64).shape[1])
    if n_smoothing_params is None:
        n_smoothing_params = (
            max(
                (int(pb.smoothing_index) for pb in compiled_penalties),
                default=-1,
            )
            + 1
        )
    model.compiled_model_ = SimpleNamespace(
        design_matrix=np.asarray(design_matrix, dtype=np.float64),
        compiled_terms=tuple(compiled_terms),
        compiled_penalties=tuple(compiled_penalties),
        metadata=dict(metadata or {}),
        n_coef=int(n_coef),
        n_smoothing_params=int(n_smoothing_params),
        predictors=(),
        predictor_full_slices=(),
    )
    return model


def test_gamma_newton_branch_exposes_distinct_working_and_fisher_weights():
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(240, 2))
    eta = 0.3 + 0.7 * np.sin(X[:, 0]) - 0.2 * X[:, 1]
    mu = np.exp(eta)
    shape = 3.0
    y = rng.gamma(shape=shape, scale=mu / shape)

    gam = GAM(k=8, family="gamma", optimize_smoothing=False, smoothing_method="fixed")
    gam.fit(X=X, y=y)

    fit_state = _fit_state(gam)
    assert fit_state is not None

    ww = np.asarray(fit_state.working_weights, dtype=np.float64)
    fw = np.asarray(fit_state.fisher_weights, dtype=np.float64)
    assert ww.shape == fw.shape
    assert np.max(np.abs(ww - fw)) > 1e-12


class _FailingStepFamily(GaussianIdentityFamily):
    def __init__(self):
        super().__init__()
        self._dev_calls = 0

    def deviance(self, y, mu, weights=None):
        self._dev_calls += 1
        if self._dev_calls >= 2:
            return np.inf
        return super().deviance(y, mu, weights=weights)


def test_pirls_step_halving_exhaustion_returns_failure_without_accepting_bad_step():
    rng = np.random.default_rng(27)
    X = rng.normal(size=(120, 2))
    y = 0.5 * np.sin(X[:, 0]) + 0.2 * X[:, 1] + rng.normal(scale=0.1, size=120)

    gam = GAM(k=8)
    gam.fit(X=X, y=y)

    X = build_full_design(_design_matrix(gam), fit_intercept=gam.fit_intercept)
    S = build_full_penalty_from_blocks(
        penalty_blocks=_penalty_blocks_seq(gam),
        smoothing_params=gam.smoothing_params,
        fit_intercept=gam.fit_intercept,
        n_coef=_n_coef(gam),
    )
    rank_rows = balanced_penalty_template_sqrt_for_rank(
        _penalty_blocks_seq(gam),
        fit_intercept=gam.fit_intercept,
        n_coef=int(_n_coef(gam)),
    )
    sol = irls_core(
        X,
        y=y,
        family=_FailingStepFamily(),
        S=S,
        max_iter=5,
        max_step_halving=0,
        offset=None,
        fit_intercept=gam.fit_intercept,
        penalty_rank_rows=rank_rows,
    )
    assert sol["failed_step"] is True
    assert sol["failure_reason"] == "step_halving_exhausted"
    assert sol["converged"] is False


def test_binomial_pirls_uses_stacked_qr_when_system_is_ill_conditioned(monkeypatch):
    x = np.linspace(-2.0, 2.0, 80, dtype=np.float64)
    y = (x > 0.0).astype(np.float64)

    gam = GAM(k=8, optimize_smoothing=False, smoothing_method="fixed")
    gam.fit(X=x[:, None], y=np.sin(x))

    called = {"stacked_qr": 0}

    def _wrapped_stacked_qr(*args, **kwargs):
        called["stacked_qr"] += 1
        return _stacked_penalized_ls_nonneg_solution(*args, **kwargs)

    irls_core_module = importlib.import_module("nampy.gam.fit.solvers.irls_core")
    monkeypatch.setattr(
        irls_core_module, "_stacked_penalized_ls_nonneg_solution", _wrapped_stacked_qr
    )
    monkeypatch.setattr(irls_core_module.np.linalg, "cond", lambda _A: 1e13)

    X = build_full_design(_design_matrix(gam), fit_intercept=gam.fit_intercept)
    S = build_full_penalty_from_blocks(
        penalty_blocks=_penalty_blocks_seq(gam),
        smoothing_params=gam.smoothing_params,
        fit_intercept=gam.fit_intercept,
        n_coef=_n_coef(gam),
    )
    rank_rows = balanced_penalty_template_sqrt_for_rank(
        _penalty_blocks_seq(gam),
        fit_intercept=gam.fit_intercept,
        n_coef=int(_n_coef(gam)),
    )
    sol = irls_core(
        X,
        y=y,
        family=BinomialLogitFamily(),
        S=S,
        max_iter=50,
        offset=None,
        fit_intercept=gam.fit_intercept,
        penalty_rank_rows=rank_rows,
    )

    assert called["stacked_qr"] > 0
    assert sol["failed_step"] is False
    assert sol["failure_reason"] is None
    assert np.all(np.isfinite(sol["coef_full"]))


def test_outer_newton_result_sets_stable_metadata():
    class _Obj:
        def __init__(self):
            self.n_fun = 0
            self.n_jac = 0
            self.n_hess = 0

        def fun(self, x):
            self.n_fun += 1
            x = np.asarray(x, dtype=np.float64)
            return float(np.sum((x - 1.0) ** 2))

        def jac(self, x):
            self.n_jac += 1
            x = np.asarray(x, dtype=np.float64)
            return 2.0 * (x - 1.0)

        def hess(self, x):
            self.n_hess += 1
            x = np.asarray(x, dtype=np.float64)
            return 2.0 * np.eye(x.size, dtype=np.float64)

    out = _optimize_outer_newton(
        objective=_Obj(),
        x0=np.array([8.0, -8.0], dtype=np.float64),
        bounds=[(-10.0, 10.0), (-10.0, 10.0)],
        max_iter=50,
    )
    assert isinstance(out, OptimizeResult)
    assert hasattr(out, "x")
    assert hasattr(out, "fun")
    assert hasattr(out, "success")
    assert hasattr(out, "message")
    assert hasattr(out, "nit")
    assert hasattr(out, "nfev")
    assert hasattr(out, "njev")
    assert hasattr(out, "nhev")
    assert np.all(np.isfinite(np.asarray(out.x, dtype=np.float64)))
    assert np.isfinite(float(out.fun))


def test_disjoint_multi_penalty_term_is_accepted_and_reparameterized():
    class _Dummy:
        pass

    class _TermBlock:
        def __init__(self, basis_train, coef_slice):
            self.basis_train = basis_train
            self.coef_slice = coef_slice

    class _PenaltyBlock:
        def __init__(self, matrix, coef_slice, smoothing_index):
            self.matrix = matrix
            self.coef_slice = coef_slice
            self.smoothing_index = smoothing_index
            self.kind = "smooth"
            self.is_null_space_penalty = False

    rng = np.random.default_rng(99)
    n = 80
    B = rng.normal(size=(n, 4))
    term_slice = slice(0, 4)

    P1 = np.diag([1.0, 1.0, 0.0, 0.0])
    P2 = np.diag([0.0, 0.0, 1.0, 1.0])

    model = _Dummy()
    model.fit_intercept = True
    model.n_samples_ = n
    term_blocks = [_TermBlock(B, term_slice)]
    penalty_blocks = [
        _PenaltyBlock(P1, term_slice, smoothing_index=0),
        _PenaltyBlock(P2, term_slice, smoothing_index=1),
    ]
    _attach_compiled_model(
        model,
        design_matrix=B,
        compiled_terms=term_blocks,
        compiled_penalties=penalty_blocks,
    )

    assert can_use_simple_ml_reml_structure(model)
    state = build_penalty_reparameterized_system(model)
    assert model.reparam_state_ is state
    assert state is not None
    assert state.Z_rand is not None
    assert state.sl_blocks is not None
    assert model.sl_blocks_ == state.sl_blocks
    assert state.Z_rand.shape[1] == 4
    assert len(state.sl_blocks) == 2
    assert state.sl_blocks[0] == SlBlock(
        term_index=0,
        repara=True,
        smoothing_index=0,
        start=0,
        stop=2,
        ncol=4,
        blockSize=2,
    )
    assert state.sl_blocks[1] == SlBlock(
        term_index=0,
        repara=True,
        smoothing_index=1,
        start=2,
        stop=4,
        ncol=4,
        blockSize=2,
    )
    groups = sl_group_indices(state)
    assert set(groups.keys()) == {0, 1}
    np.testing.assert_array_equal(groups[0], np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(groups[1], np.array([2, 3], dtype=np.int64))


def test_overlapping_null_space_penalties_on_one_term_are_accepted():
    class _Dummy:
        pass

    class _TermBlock:
        def __init__(self, basis_train, coef_slice):
            self.basis_train = basis_train
            self.coef_slice = coef_slice

    class _PenaltyBlock:
        def __init__(self, matrix, coef_slice, smoothing_index, *, is_null=False):
            self.matrix = matrix
            self.coef_slice = coef_slice
            self.smoothing_index = smoothing_index
            self.kind = "smooth"
            self.is_null_space_penalty = is_null

    rng = np.random.default_rng(199)
    n = 80
    B = rng.normal(size=(n, 4))
    term_slice = slice(0, 4)

    P = np.diag([1.0, 1.0, 0.0, 0.0])
    N1 = np.diag([0.0, 0.0, 1.0, 1.0])
    N2 = np.diag([0.0, 0.0, 1.0, 1.0])

    model = _Dummy()
    model.fit_intercept = True
    model.n_samples_ = n
    term_blocks = [_TermBlock(B, term_slice)]
    penalty_blocks = [
        _PenaltyBlock(P, term_slice, smoothing_index=0),
        _PenaltyBlock(N1, term_slice, smoothing_index=1, is_null=True),
        _PenaltyBlock(N2, term_slice, smoothing_index=2, is_null=True),
    ]
    _attach_compiled_model(
        model,
        design_matrix=B,
        compiled_terms=term_blocks,
        compiled_penalties=penalty_blocks,
    )

    assert can_use_simple_ml_reml_structure(model)
    state = build_penalty_reparameterized_system(model)
    assert state is not None
    assert state.Z_rand is not None
    assert state.sl_blocks is not None
    assert state.Z_rand.shape[1] == 6
    assert len(state.sl_blocks) == 3
    assert [b.smoothing_index for b in state.sl_blocks] == [0, 1, 2]
    assert [b.blockSize for b in state.sl_blocks] == [2, 2, 2]
    groups = sl_group_indices(state)
    assert set(groups.keys()) == {0, 1, 2}
    np.testing.assert_array_equal(groups[0], np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(groups[1], np.array([2, 3], dtype=np.int64))
    np.testing.assert_array_equal(groups[2], np.array([4, 5], dtype=np.int64))


def test_dynamic_reparam_design_depends_on_current_sp():
    class _Dummy:
        pass

    class _TermBlock:
        def __init__(self, basis_train, coef_slice):
            self.basis_train = basis_train
            self.coef_slice = coef_slice

    class _PenaltyBlock:
        def __init__(self, matrix, coef_slice, smoothing_index):
            self.matrix = matrix
            self.coef_slice = coef_slice
            self.smoothing_index = smoothing_index
            self.kind = "smooth"
            self.is_null_space_penalty = False

    rng = np.random.default_rng(101)
    n = 60
    B = rng.normal(size=(n, 4))
    term_slice = slice(0, 4)

    model = _Dummy()
    model.fit_intercept = True
    model.n_samples_ = n
    term_blocks = [_TermBlock(B, term_slice)]
    penalty_blocks = [
        _PenaltyBlock(np.diag([1.0, 1.0, 0.0, 0.0]), term_slice, smoothing_index=0),
        _PenaltyBlock(np.diag([0.0, 0.0, 1.0, 1.0]), term_slice, smoothing_index=1),
    ]
    _attach_compiled_model(
        model,
        design_matrix=B,
        compiled_terms=term_blocks,
        compiled_penalties=penalty_blocks,
        n_coef=4,
        n_smoothing_params=2,
    )
    X = build_full_design(B, fit_intercept=True)

    d0 = dynamic_reparam_design(model, X, np.array([0.5, 2.0], dtype=np.float64))
    d1 = dynamic_reparam_design(model, X, np.array([2.0, 0.5], dtype=np.float64))

    assert d0.X_fix.shape == d1.X_fix.shape == (n, 1)
    assert d0.Z_rand.shape == d1.Z_rand.shape == (n, 4)
    assert np.isfinite(d0.penalty_logdet)
    assert np.isfinite(d1.penalty_logdet)
    assert np.max(np.abs(d0.Z_rand - d1.Z_rand)) > 1e-8


def test_canonical_gam_reparam_state_matches_dynamic_design():
    rng = np.random.default_rng(102)
    n = 50
    B = rng.normal(size=(n, 4))
    x_full = build_full_design(B, fit_intercept=True)

    class _Dummy:
        pass

    class _TermBlock:
        def __init__(self, basis_train, coef_slice):
            self.basis_train = basis_train
            self.coef_slice = coef_slice

    class _PenaltyBlock:
        def __init__(self, matrix, coef_slice, smoothing_index):
            self.matrix = matrix
            self.coef_slice = coef_slice
            self.smoothing_index = smoothing_index
            self.kind = "smooth"
            self.is_null_space_penalty = False

    term_slice = slice(0, 4)
    model = _Dummy()
    model.fit_intercept = True
    model.n_samples_ = n
    term_blocks = [_TermBlock(B, term_slice)]
    penalty_blocks = [
        _PenaltyBlock(np.diag([1.0, 1.0, 0.0, 0.0]), term_slice, smoothing_index=0),
        _PenaltyBlock(np.diag([0.0, 0.0, 1.0, 1.0]), term_slice, smoothing_index=1),
    ]
    _attach_compiled_model(
        model,
        design_matrix=B,
        compiled_terms=term_blocks,
        compiled_penalties=penalty_blocks,
        n_coef=4,
        n_smoothing_params=2,
    )

    sp = np.array([0.5, 2.0], dtype=np.float64)
    canonical = build_gam_fit3_reparam_state(model, x_full, sp, deriv=1)
    dynamic = dynamic_reparam_design(model, x_full, sp)

    assert canonical.U1.shape == (5, 5)
    assert canonical.T.shape == (5, 5)
    assert canonical.St.shape == (5, 5)
    assert canonical.Sr.shape[1] == 5
    assert canonical.Mp == 1
    np.testing.assert_allclose(canonical.X_fix, dynamic.X_fix, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(canonical.Z_rand, dynamic.Z_rand, rtol=0.0, atol=1e-12)


def test_gam_fit3_state_groups_shared_id_from_exact_estimate_setup():
    data = _make_gaussian_data(seed=777, n=120)
    gam = GAM(
        family="gaussian",
        formula='y ~ s(x0, bs="cr", k=8, id="g") + s(x1, bs="cr", k=8, id="g")',
        optimize_smoothing=False,
        smoothing_method="REML",
    )
    gam.fit(data=data)

    estimate = build_estimate_gam_setup_state(gam)
    state = build_gam_fit3_reparam_state(
        gam,
        np.asarray(estimate.X, dtype=np.float64),
        np.asarray(gam.smoothing_params, dtype=np.float64),
        deriv=1,
    )
    dynamic = dynamic_reparam_design(
        gam,
        np.asarray(estimate.X, dtype=np.float64),
        np.asarray(gam.smoothing_params, dtype=np.float64),
    )

    penalty_blocks = list(_penalty_blocks_seq(gam))
    assert len(penalty_blocks) == 2
    assert {int(pb.smoothing_index) for pb in penalty_blocks} == {0}
    grouped_gram = (
        np.asarray(state.UrS[0], dtype=np.float64)
        @ np.asarray(state.UrS[0], dtype=np.float64).T
    )
    exact_block_gram = sum(
        np.asarray(root, dtype=np.float64) @ np.asarray(root, dtype=np.float64).T
        for root in estimate.UrS[: len(penalty_blocks)]
    )

    assert state.Mp == estimate.Mp
    np.testing.assert_allclose(grouped_gram, exact_block_gram, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(state.X_fix, dynamic.X_fix, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(state.Z_rand, dynamic.Z_rand, rtol=0.0, atol=1e-12)


def test_t2_term_emits_one_sl_block_per_penalty_slice():
    rng = np.random.default_rng(123)
    n = 80
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    y = np.sin(x0) + 0.3 * x1**2
    data = pd.DataFrame({"y": y, "x0": x0, "x1": x1})

    gam = GAM(
        family="gaussian",
        formula='y ~ t2(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3, 0.9])',
        optimize_smoothing=False,
        smoothing_method="fixed",
    )
    gam.fit(data=data)

    state = gam.reparam_state_
    assert state is not None
    assert state.sl_blocks is not None
    assert len(state.sl_blocks) == 3
    assert [b.blockSize for b in state.sl_blocks] == [9, 6, 6]
    assert [(b.start, b.stop) for b in state.sl_blocks] == [(0, 9), (9, 15), (15, 21)]
    assert [b.ncol for b in state.sl_blocks] == [24, 24, 24]
    assert all(b.repara for b in state.sl_blocks)
    assert [b.term_index for b in state.sl_blocks] == [0, 0, 0]
    assert [b.smoothing_index for b in state.sl_blocks] == [0, 1, 2]
    groups = sl_group_indices(state)
    assert set(groups.keys()) == {0, 1, 2}
    np.testing.assert_array_equal(groups[0], np.arange(0, 9, dtype=np.int64))
    np.testing.assert_array_equal(groups[1], np.arange(9, 15, dtype=np.int64))
    np.testing.assert_array_equal(groups[2], np.arange(15, 21, dtype=np.int64))


def test_mrf_term_emits_single_sl_block():
    data = _make_mrf_data()
    gam = GAM(
        family="gaussian",
        formula=(
            'y ~ s(region, bs="mrf", k=3, '
            'xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B"))))'
        ),
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=np.array([0.8]),
    )
    gam.fit(data=data)

    state = gam.reparam_state_
    assert state is not None
    assert state.sl_blocks is not None
    assert len(state.sl_blocks) == 1
    block = state.sl_blocks[0]
    assert block.repara is True
    assert block.term_index == 0
    assert block.smoothing_index == 0
    assert (block.start, block.stop, block.blockSize) == (0, 2, 2)
    groups = sl_group_indices(state)
    assert set(groups.keys()) == {0}
    np.testing.assert_array_equal(groups[0], np.array([0, 1], dtype=np.int64))


def test_t2_ts_cr_predict_matrix_preserves_penalized_blocks():
    data = _make_gaussian_data(seed=375, n=180)
    gam = GAM(
        family="gaussian",
        formula='y ~ t2(x0, x1, bs=["ts", "cr"], k=[6, 6])',
        optimize_smoothing=True,
        smoothing_method="REML",
    )
    gam.fit(data=data)

    term = gam.compiled_model_.compiled_terms[0]
    X_new = np.asarray(data[["x0", "x1"]], dtype=np.float64)
    B_new = term.predict_matrix(X_new)

    assert B_new.ndim == 2
    assert B_new.shape[0] == len(data)
    assert B_new.shape[1] == term.basis_train.shape[1]


def test_pirls_laplace_reml_derivatives_dispatch_to_exact_backend(monkeypatch):
    x = np.linspace(-2.0, 2.0, 40, dtype=np.float64)
    y = (x > 0.0).astype(np.float64)

    gam = GAM(
        k=8, family="binomial", optimize_smoothing=False, smoothing_method="fixed"
    )
    gam.fit(X=x[:, None], y=y)

    seen = {"grad": 0, "hess": 0, "fd_grad": 0, "fd_hess": 0}

    def _grad_stub(model, y, log_sp, method="REML"):
        del model, y, log_sp, method
        seen["grad"] += 1
        return np.array([123.0], dtype=np.float64)

    def _hess_stub(model, y, log_sp, method="REML"):
        del model, y, log_sp, method
        seen["hess"] += 1
        return np.array([[456.0]], dtype=np.float64)

    def _fd_grad(*args, **kwargs):
        del args, kwargs
        seen["fd_grad"] += 1
        return np.array([-1.0], dtype=np.float64)

    def _fd_hess(*args, **kwargs):
        del args, kwargs
        seen["fd_hess"] += 1
        return np.array([[-1.0]], dtype=np.float64)

    monkeypatch.setattr(
        criteria_dispatch, "criterion_gradient_ml_reml_pirls_exact", _grad_stub
    )
    monkeypatch.setattr(
        criteria_dispatch, "criterion_hessian_ml_reml_pirls_exact", _hess_stub
    )
    monkeypatch.setattr(criteria_dispatch, "criterion_gradient_numerical", _fd_grad)
    monkeypatch.setattr(criteria_dispatch, "criterion_hessian_numerical", _fd_hess)

    grad = criteria_dispatch.criterion_gradient(
        gam, y, np.log(gam.smoothing_params), method="reml"
    )
    hess = criteria_dispatch.criterion_hessian(
        gam, y, np.log(gam.smoothing_params), method="reml"
    )

    np.testing.assert_array_equal(grad, np.array([123.0], dtype=np.float64))
    np.testing.assert_array_equal(hess, np.array([[456.0]], dtype=np.float64))
    assert seen == {"grad": 1, "hess": 1, "fd_grad": 0, "fd_hess": 0}


def test_direct_exact_pirls_derivative_entrypoint_runs_on_canonical_reparam_state():
    x = np.linspace(-2.0, 2.0, 40, dtype=np.float64)
    y = (x > 0.0).astype(np.float64)

    gam = GAM(
        k=8, family="binomial", optimize_smoothing=False, smoothing_method="fixed"
    )
    gam.fit(X=x[:, None], y=y)

    grad = criterion_gradient_ml_reml_pirls_exact(
        gam, y, np.log(gam.smoothing_params), "REML"
    )
    assert grad.shape == (1,)
    assert np.all(np.isfinite(grad))


def test_tensor_id_metadata_maps_one_smoothing_id_to_multiple_sp_indices():
    rng = np.random.default_rng(321)
    n = 80
    x0 = rng.uniform(-1.0, 1.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    y = np.sin(x0) + x1**2
    data = pd.DataFrame({"y": y, "x0": x0, "x1": x1})

    gam = GAM(
        family="gaussian",
        formula='y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5], id="g")',
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=[0.8, 1.2],
    )
    gam.fit(data=data)

    compiled = gam.compiled_model_
    assert compiled is not None

    mapping = compiled.metadata.get("s_id_to_sp_indices", {})
    assert mapping == {"g": [0, 1]}
    assert compiled.n_smoothing_params == 2

    group_specs = compiled.metadata.get("penalty_group_specs", [])
    assert len(group_specs) == 1
    group = group_specs[0]
    assert group.smoothing_id == "g"
    assert group.sp_count == 2
    assert group.sp_indices == [0, 1]


def test_tensor_id_smoothing_param_mapping_accepts_id_keyed_values():
    rng = np.random.default_rng(654)
    n = 90
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.0, 1.0, size=n)
    y = np.cos(x0) + 0.5 * x1
    data = pd.DataFrame({"y": y, "x0": x0, "x1": x1})

    gam = GAM(
        family="gaussian",
        formula='y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5], id="g")',
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params={"g": [0.7, 1.3]},
    )
    gam.fit(data=data)

    np.testing.assert_allclose(
        np.asarray(gam.smoothing_params, dtype=np.float64),
        np.array([0.7, 1.3], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
