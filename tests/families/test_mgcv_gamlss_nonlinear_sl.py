from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose

from nampy.gam.compiler.structures import (
    CompiledModel,
    CompiledPenalty,
    CompiledPredictor,
    CompiledTerm,
)
from nampy.gam.fit.solvers.general_family.fixed_smoothing import (
    build_general_family_setup_state,
    run_general_family_fixed_smoothing,
    solve_general_family_fit,
)
from nampy.gam.fit.solvers.general_family.newton import (
    GeneralNewtonControl,
    _sl_ldetS,
    _sl_mult,
    _sl_second_mult,
    _sl_term_mult,
    postprocess_general_newton_fit,
    solve_general_newton_fit,
)


@dataclass
class _NonlinearSlBlock:
    start: int = 1
    stop: int = 2
    linear: bool = False
    repara: bool = False
    n_sp: int = 2
    S: list[np.ndarray] = field(
        default_factory=lambda: [
            np.eye(2, dtype=np.float64),
            np.eye(2, dtype=np.float64),
        ]
    )
    lambda_: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    _diag: np.ndarray = field(default_factory=lambda: np.ones(2, dtype=np.float64))

    def updateS(self, rho, block=None):
        block = self if block is None else block
        rho = np.asarray(rho, dtype=np.float64)
        block.lambda_ = rho.copy()
        block._diag = np.array([np.exp(rho[0]) + 1.0, np.exp(rho[1]) + 1.0])
        return block

    def ldS(self, block=None, deriv=2):
        block = self if block is None else block
        e = np.exp(np.asarray(block.lambda_, dtype=np.float64))
        d = np.asarray(block._diag, dtype=np.float64)
        out = {
            "ldS": float(np.sum(np.log(d))),
            "ldS1": e / d,
            "ldS2": np.diag(e / (d**2)),
        }
        if deriv < 2:
            out["ldS2"] = np.zeros((2, 2), dtype=np.float64)
        return out

    def St(self, block=None, mode=2):
        block = self if block is None else block
        diag = np.asarray(block._diag, dtype=np.float64)
        mat = np.diag(diag)
        root = np.diag(np.sqrt(diag))
        if mode == 2:
            return {"E": root, "S": mat}
        if mode == 1:
            return {"E": root}
        return {"S": mat}

    def AS(self, A, block=None):
        block = self if block is None else block
        return np.asarray(A, dtype=np.float64) @ np.diag(np.asarray(block._diag))

    def AdS(self, A, block=None, i=1, j=None):
        block = self if block is None else block
        if isinstance(block, (int, np.integer)):
            i = int(block)
            block = self
        A = np.asarray(A, dtype=np.float64)
        first = [
            np.diag([np.exp(block.lambda_[0]), 0.0]),
            np.diag([0.0, np.exp(block.lambda_[1])]),
        ]
        if j is None:
            return A @ first[int(i) - 1]
        if int(i) != int(j):
            return np.zeros_like(A)
        return A @ first[int(i) - 1]


class _QuadraticFamily:
    def initialize(self, y, X, jj, offset=None, weights=None, E=None):  # noqa: ARG002
        return np.zeros(X.shape[1], dtype=np.float64)

    def predict(self, eta):
        return np.asarray(eta, dtype=np.float64)

    def ll(
        self,
        y,
        X,
        jj,  # noqa: ARG002
        coef,
        weights,
        offset=None,  # noqa: ARG002
        deriv=0,
        d1b=None,
        d2b=None,  # noqa: ARG002
        fh=None,  # noqa: ARG002
        D=None,  # noqa: ARG002
    ):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        coef = np.asarray(coef, dtype=np.float64)
        weights = (
            np.ones(y.shape[0], dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        resid = y - X @ coef
        WX = X * weights[:, np.newaxis]
        out = {"l": float(-0.5 * np.sum(weights * resid * resid))}
        if deriv > 0:
            out["lb"] = X.T @ (weights * resid)
            out["lbb"] = -(X.T @ WX)
        if deriv == 2 and d1b is not None:
            out["d1H"] = np.zeros(d1b.shape[1], dtype=np.float64)
        elif deriv >= 3 and d1b is not None:
            out["d1H"] = [
                np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
            ] * d1b.shape[1]
        if deriv >= 4 and d1b is not None:
            m = d1b.shape[1]
            out["trHid2H"] = np.zeros(m * (m + 1) // 2, dtype=np.float64)
        return out


class _AnalyticQuadraticFamily(_QuadraticFamily):
    supports_analytic_outer_derivatives = True


def _nonlinear_sl():
    return SimpleNamespace(
        blocks=[_NonlinearSlBlock()],
        S=np.eye(2, dtype=np.float64),
    )


def _term_owned_nonlinear_sl(_term, _penalty_indices, start, stop):
    return _NonlinearSlBlock(start=int(start), stop=int(stop))


def _compiled_general_family_model_with_nonlinear_sl(*, family=None):
    design = np.eye(2, dtype=np.float64)
    term = CompiledTerm(
        label="s(x)",
        coef_slice=slice(0, 2),
        basis_train=design,
        smoothing_indices=[0, 1],
        smoothing_ids=["sp0", "sp1"],
        n_penalties=2,
        term_type="smooth",
        basis_name="test",
        term_id="term0",
        smoothing_group_id="term0",
        metadata={"general_family_nonlinear_sl": _term_owned_nonlinear_sl},
    )
    penalties = (
        CompiledPenalty(
            label="s(x)",
            coef_slice=slice(0, 2),
            matrix=np.eye(2, dtype=np.float64),
            smoothing_index=0,
            term_index=0,
            smoothing_id="sp0",
        ),
        CompiledPenalty(
            label="s(x)",
            coef_slice=slice(0, 2),
            matrix=np.eye(2, dtype=np.float64),
            smoothing_index=1,
            term_index=0,
            smoothing_id="sp1",
        ),
    )
    predictor = CompiledPredictor(
        name="eta1",
        design_matrix=design,
        compiled_terms=(term,),
        compiled_penalties=penalties,
        smoothing_parameter_map={"sp0": 0, "sp1": 1},
        n_coef=2,
        n_smoothing_params=2,
        has_intercept=False,
    )
    compiled_model = CompiledModel(
        predictors=(predictor,),
        design_matrix=design,
        compiled_terms=(term,),
        compiled_penalties=penalties,
        metadata={},
        n_coef=2,
        n_smoothing_params=2,
        predictor_full_slices=(slice(0, 2),),
        coef_reduced_to_full_idx=np.arange(2, dtype=int),
    )
    return SimpleNamespace(
        n_samples_=2,
        max_irls_iter=50,
        irls_tol=1e-12,
        hparams={},
        prior_weights_=np.ones(2, dtype=np.float64),
        compiled_model_=compiled_model,
        family=_QuadraticFamily() if family is None else family,
        _optim_method="REML",
    )


def test_nonlinear_sl_helpers_match_expected_penalty_derivatives():
    """Verify that nonlinear sl helpers match expected penalty derivatives."""
    rho = np.log(np.array([2.0, 3.0], dtype=np.float64))
    state = _sl_ldetS(
        _nonlinear_sl(),
        rho=rho,
        fixed=np.zeros(2, dtype=bool),
        np_=2,
        root=True,
        Stot=True,
        deriv=2,
    )

    expected_diag = np.array([3.0, 4.0], dtype=np.float64)
    expected_d1 = np.array([2.0 / 3.0, 3.0 / 4.0], dtype=np.float64)
    expected_d2 = np.diag([2.0 / 9.0, 3.0 / 16.0]).astype(np.float64)
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    assert_allclose(state["ldetS"], float(np.log(3.0) + np.log(4.0)), atol=1e-12)
    assert_allclose(state["ldet1"], expected_d1, atol=1e-12)
    assert_allclose(state["ldet2"], expected_d2, atol=1e-12)
    assert_allclose(state["S"], np.diag(expected_diag), atol=1e-12)
    assert_allclose(state["E"], np.diag(np.sqrt(expected_diag)), atol=1e-12)
    assert_allclose(
        _sl_mult(state["Sl"], A, 0, full=True),
        np.diag(expected_diag) @ A,
        atol=1e-12,
    )
    assert_allclose(
        _sl_mult(state["Sl"], A, 1, full=True),
        np.diag([2.0, 0.0]) @ A,
        atol=1e-12,
    )
    assert_allclose(
        _sl_mult(state["Sl"], A, 2, full=True),
        np.diag([0.0, 3.0]) @ A,
        atol=1e-12,
    )
    terms = _sl_term_mult(state["Sl"], A, full=True)
    assert len(terms) == 2
    assert_allclose(terms[0], np.diag([2.0, 0.0]) @ A, atol=1e-12)
    assert_allclose(terms[1], np.diag([0.0, 3.0]) @ A, atol=1e-12)
    assert_allclose(
        _sl_second_mult(state["Sl"], A, 1, 1, full=True),
        np.diag([2.0, 0.0]) @ A,
        atol=1e-12,
    )
    assert_allclose(
        _sl_second_mult(state["Sl"], A, 1, 2, full=True),
        np.zeros_like(A),
        atol=1e-12,
    )


def test_solve_general_newton_fit_accepts_nonlinear_sl_blocks():
    """Verify that solve general newton fit accepts nonlinear sl blocks."""
    rho = np.log(np.array([2.0, 3.0], dtype=np.float64))
    fit = solve_general_newton_fit(
        np.eye(2, dtype=np.float64),
        np.array([1.0, -1.0], dtype=np.float64),
        [np.array([0, 1], dtype=int)],
        rho,
        np.zeros((2, 2), dtype=np.float64),
        [np.zeros((2, 2), dtype=np.float64) for _ in range(2)],
        ldetS=0.0,
        ldetS1=np.zeros(2, dtype=np.float64),
        ldetS2=np.zeros((2, 2), dtype=np.float64),
        family=_QuadraticFamily(),
        weights=None,
        offset=None,
        deriv=2,
        control=GeneralNewtonControl(maxit=50, epsilon=1e-12, trace=False),
        Mp=0,
        Sl=_nonlinear_sl(),
    )

    assert_allclose(
        fit["coef"],
        np.array([0.25, -0.2], dtype=np.float64),
        atol=1e-10,
    )
    assert fit["REML1"].shape == (2,)
    assert fit["REML2"].shape == (2, 2)


def test_general_family_setup_state_materializes_term_owned_nonlinear_sl():
    """Verify that general family setup state materializes term owned nonlinear sl."""
    model = _compiled_general_family_model_with_nonlinear_sl()
    smoothing_params = np.array([2.0, 3.0], dtype=np.float64)

    setup = build_general_family_setup_state(model, smoothing_params)

    assert len(setup.Sl) == 1
    assert not bool(getattr(setup.Sl[0], "linear", True))
    assert tuple(getattr(setup.Sl[0], "penalty_indices", ())) == (0, 1)
    assert_allclose(setup.St, np.diag([3.0, 4.0]), atol=1e-12)
    assert_allclose(
        setup.ldetS1,
        np.array([2.0 / 3.0, 3.0 / 4.0], dtype=np.float64),
        atol=1e-12,
    )
    assert_allclose(
        setup.ldetS2,
        np.diag([2.0 / 9.0, 3.0 / 16.0]).astype(np.float64),
        atol=1e-12,
    )
    assert len(setup.S_blocks) == 0


def test_run_general_family_fixed_smoothing_accepts_model_generated_nonlinear_sl():
    """
    Verify that run general family fixed smoothing accepts model generated nonlinear sl.
    """
    model = _compiled_general_family_model_with_nonlinear_sl()
    run = run_general_family_fixed_smoothing(
        model,
        np.array([1.0, -1.0], dtype=np.float64),
        np.array([2.0, 3.0], dtype=np.float64),
        deriv=2,
    )

    assert_allclose(run["setup"].St, np.diag([3.0, 4.0]), atol=1e-12)
    assert_allclose(
        run["fit"]["coef"],
        np.array([0.25, -0.2], dtype=np.float64),
        atol=1e-10,
    )
    assert run["fit"]["REML1"].shape == (2,)
    assert run["fit"]["REML2"].shape == (2, 2)


def test_solve_general_family_fit_accepts_model_generated_nonlinear_sl():
    """Verify that solve general family fit accepts model generated nonlinear sl."""
    model = _compiled_general_family_model_with_nonlinear_sl()
    sol = solve_general_family_fit(
        model,
        np.array([1.0, -1.0], dtype=np.float64),
        np.array([2.0, 3.0], dtype=np.float64),
    )

    assert_allclose(sol.beta, np.array([0.25, -0.2], dtype=np.float64), atol=1e-10)
    assert_allclose(sol.penalty_matrix, np.diag([3.0, 4.0]), atol=1e-12)


def test_solve_general_family_fit_applies_exact_nonlinear_vb_corr():
    """Verify that solve general family fit applies exact nonlinear VB corr."""
    model = _compiled_general_family_model_with_nonlinear_sl(
        family=_AnalyticQuadraticFamily()
    )
    y = np.array([1.0, -1.0], dtype=np.float64)
    sp = np.array([2.0, 3.0], dtype=np.float64)

    run = run_general_family_fixed_smoothing(model, y, sp, deriv=2)
    setup = run["setup"]
    fit = run["fit"]
    without_exact = postprocess_general_newton_fit(
        fit,
        Sl=setup.Sl,
        L_map=None,
        lsp0=None,
        S_blocks=setup.S_blocks,
        off=None,
        smoothing_params=setup.smoothing_params,
    )
    with_exact = postprocess_general_newton_fit(
        fit,
        Sl=setup.Sl,
        L_map=None,
        lsp0=None,
        S_blocks=setup.S_blocks,
        off=None,
        smoothing_params=setup.smoothing_params,
        penalty_matrix=setup.St,
        penalty_derivatives=setup.penalty_derivatives,
    )

    sol = solve_general_family_fit(model, y, sp)

    assert float(np.trace(with_exact["Vc"])) > float(np.trace(without_exact["Vc"]))
    assert_allclose(sol.cov_unconditional, with_exact["Vc"], atol=1e-12)
