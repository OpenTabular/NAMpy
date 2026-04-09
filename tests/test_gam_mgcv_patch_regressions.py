import numpy as np
import pandas as pd
from mgcv_parity_utils import _make_mrf_data
from scipy.optimize import OptimizeResult

from nampy.basemodels.gam import GAM
from nampy.gam.families.exponential import GaussianIdentityFamily
from nampy.gam.fit.solvers.pirls_core import fit_pirls_core
from nampy.gam.smoothing_selection.criteria import dispatch as criteria_dispatch
from nampy.gam.smoothing_selection.optimize import (
    _rollback_working_infinite_smoothing_params,
)
from nampy.gam.smoothing_selection.reparam import (
    SlBlock,
    build_penalty_reparameterized_system,
    can_use_simple_ml_reml_structure,
    sl_group_indices,
)


def test_gamma_newton_branch_exposes_distinct_working_and_fisher_weights():
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(240, 2))
    eta = 0.3 + 0.7 * np.sin(X[:, 0]) - 0.2 * X[:, 1]
    mu = np.exp(eta)
    shape = 3.0
    y = rng.gamma(shape=shape, scale=mu / shape)

    gam = GAM(k=8, family="gamma", optimize_smoothing=False, smoothing_method="fixed")
    gam.fit(X=X, y=y)

    ww = np.asarray(gam.fit_state_.working_weights, dtype=np.float64)
    fw = np.asarray(gam.fit_state_.fisher_weights, dtype=np.float64)
    assert ww.shape == fw.shape
    assert np.max(np.abs(ww - fw)) > 1e-12


class _FailingStepFamily(GaussianIdentityFamily):
    def __init__(self):
        super().__init__()
        self._dev_calls = 0

    def deviance(self, y, mu):
        self._dev_calls += 1
        if self._dev_calls >= 2:
            return np.inf
        return super().deviance(y, mu)


def test_pirls_step_halving_exhaustion_returns_failure_without_accepting_bad_step():
    rng = np.random.default_rng(27)
    X = rng.normal(size=(120, 2))
    y = 0.5 * np.sin(X[:, 0]) + 0.2 * X[:, 1] + rng.normal(scale=0.1, size=120)

    gam = GAM(k=8)
    gam.fit(X=X, y=y)

    sol = fit_pirls_core(
        Z=gam.Z,
        y=y,
        penalty_blocks=gam.penalty_blocks_,
        smoothing_params=gam.smoothing_params,
        family=_FailingStepFamily(),
        fit_intercept=gam.fit_intercept,
        max_iter=5,
        max_step_halving=0,
        offset=None,
    )
    assert sol["failed_step"] is True
    assert sol["failure_reason"] == "step_halving_exhausted"
    assert sol["converged"] is False


def test_optimizer_rollback_sets_stable_metadata(monkeypatch):
    class _Obj:
        def __init__(self):
            self.model = object()
            self.y = np.array([0.0])
            self.use_gradient = True

        def fun(self, x):
            x = np.asarray(x, dtype=np.float64)
            return float(np.sum((x - 1.0) ** 2))

        def jac(self, x):
            x = np.asarray(x, dtype=np.float64)
            return 2.0 * (x - 1.0)

        def hess(self, x):
            x = np.asarray(x, dtype=np.float64)
            return 2.0 * np.eye(x.size, dtype=np.float64)

    def _signal(_model, _y, x, method):
        del _model, _y, x, method
        return np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)

    monkeypatch.setattr(criteria_dispatch, "criterion_infinite_sp_signal", _signal)

    result = OptimizeResult(
        x=np.array([8.0, -8.0], dtype=np.float64),
        fun=98.0,
        success=True,
        message="ok",
    )
    out = _rollback_working_infinite_smoothing_params(
        objective=_Obj(),
        result=result,
        x0=np.array([0.0, 0.0], dtype=np.float64),
        bounds=[(-10.0, 10.0), (-10.0, 10.0)],
        method="reml",
    )
    assert getattr(out, "rolled_back_infinite_sp", False) is True
    assert hasattr(out, "rollback_start_x")
    assert hasattr(out, "rollback_final_x")


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
    model.design_ = object()
    model.fit_intercept = True
    model.n_samples_ = n
    model.term_blocks_ = [_TermBlock(B, term_slice)]
    model.penalty_blocks_ = [
        _PenaltyBlock(P1, term_slice, smoothing_index=0),
        _PenaltyBlock(P2, term_slice, smoothing_index=1),
    ]

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

    mapping = gam.design_.metadata.get("s_id_to_sp_indices", {})
    assert mapping == {"g": [0, 1]}
    assert gam.n_smoothing_params_ == 2

    group_specs = gam.design_.metadata.get("penalty_group_specs", [])
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
