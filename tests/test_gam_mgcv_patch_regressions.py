import numpy as np
from scipy.optimize import OptimizeResult

from nampy.basemodels.gam import GAM
from nampy.gam.families.exponential import GaussianIdentityFamily
from nampy.gam.fit.solvers.pirls_core import fit_pirls_core
from nampy.gam.smoothing_selection import optimize as optimize_mod
from nampy.gam.smoothing_selection.optimize import _rollback_working_infinite_smoothing_params
from nampy.gam.smoothing_selection.criteria import dispatch as criteria_dispatch
from nampy.gam.smoothing_selection.reparam import (
    build_penalty_reparameterized_system,
    can_use_simple_ml_reml_structure,
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
    build_penalty_reparameterized_system(model)
    assert model.Z_rand_.shape[1] == 4
    assert set(model._reparam_sp_groups_.keys()) == {0, 1}
