import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult

from nampy.gam import GAM
from nampy.gam._model_state import (
    _design_matrix,
    _fit_state,
    _n_coef,
    _penalty_blocks_seq,
)
from nampy.gam.compiler.factory import instantiate_term
from nampy.gam.families import BinomialLogitFamily, GaussianIdentityFamily
from nampy.gam.fit.linalg.stacked_qr import (
    STACKED_QR_RANK_TOLERANCE,
    _dgeqp3_economic_r,
    _get_r_pqr_serial,
    _scatter_pivoted_rank_matrix_to_full,
    _stacked_penalized_ls_nonneg_solution,
    balanced_penalty_template_sqrt_for_rank,
    build_penalized_qr_state_nonnegative,
    penalty_sqrt_rows,
    solve_gaussian_penalized_ls_stacked_qr,
    stacked_qr_covariance_from_factor,
)
from nampy.gam.fit.penalized_system import (
    build_full_design,
    build_full_penalty_from_blocks,
)
from nampy.gam.fit.solvers.irls_core import irls_core
from nampy.gam.fit.state import FitCoreSolution, FitState, assign_fit_solution
from nampy.gam.formula import extract_formula_terms, parse_gam_formula
from nampy.gam.results import FitResult
from nampy.gam.smoothing_selection.criteria import dispatch as criteria_dispatch
from nampy.gam.smoothing_selection.criteria import gaussian as gaussian_criteria
from nampy.gam.smoothing_selection.criteria import pirls as pirls_criteria
from nampy.gam.smoothing_selection.criteria.gaussian import criterion_ml_reml_exact
from nampy.gam.smoothing_selection.criteria.gaussian_reml_algebra import (
    pearson_method_scale_estimate,
    profiled_gaussian_reml_variance,
)
from nampy.gam.smoothing_selection.criteria.pirls.derivatives import (
    criterion_gradient_ml_reml_pirls_exact,
)
from nampy.gam.smoothing_selection.optimize.newton import _optimize_outer_newton
from nampy.gam.smoothing_selection.optimize.newton_strict import (
    _optimize_outer_newton_strict,
)
from nampy.gam.smoothing_selection.reparam import (
    SlBlock,
    build_estimate_gam_setup_state,
    build_penalty_reparameterization_state,
    build_penalty_reparameterized_system,
    can_use_simple_ml_reml_structure,
    dynamic_reparam_design,
    sl_group_indices,
)
from nampy.gam.specs.build import build_formula_model
from nampy.splines.univariate.tp import construct_tprs_basis
from tests.mgcv_parity_utils import _make_gaussian_data


def test_gcv_scores_square_negative_denominator_like_mgcv(monkeypatch):
    model = SimpleNamespace(n_samples_=10, score_gamma=2.0)
    sol = {"trace_H": 10.0, "rss": 5.0, "deviance": 7.0}

    monkeypatch.setattr(
        gaussian_criteria,
        "expand_smoothing_params_from_log",
        lambda model, log_sp: np.array([1.0], dtype=np.float64),
    )
    monkeypatch.setattr(
        gaussian_criteria,
        "solve_gaussian_given_smoothing",
        lambda model, y, sp: sol,
    )
    monkeypatch.setattr(
        pirls_criteria,
        "expand_smoothing_params_from_log",
        lambda model, log_sp: np.array([1.0], dtype=np.float64),
    )
    monkeypatch.setattr(
        pirls_criteria,
        "solve_pirls_given_smoothing",
        lambda model, y, sp: sol,
    )

    y = np.zeros(10, dtype=np.float64)
    log_sp = np.zeros(1, dtype=np.float64)

    assert gaussian_criteria.gcv_score_gaussian(model, y, log_sp) == pytest.approx(0.5)
    assert pirls_criteria.criterion_gcv_pirls(model, y, log_sp) == pytest.approx(0.7)


def test_pearson_fletcher_nan_correction_is_not_clamped_finite():
    scale = pearson_method_scale_estimate(
        12.0,
        2.0,
        8.0,
        dev_extra=3.0,
        fletcher=True,
        y=np.array([1.0, 2.0], dtype=np.float64),
        mu=np.array([1.0, 2.0], dtype=np.float64),
        dvar_over_var=np.array([np.nan, np.nan], dtype=np.float64),
    )

    assert scale == pytest.approx((12.0 + 3.0) / (8.0 - 2.0))


def test_profiled_gaussian_variance_uses_mgcv_ml_and_reml_denominators():
    weights = np.array([1.0, 0.0, 2.0, 3.0], dtype=np.float64)

    ml = profiled_gaussian_reml_variance(
        10.0,
        2.0,
        4.0,
        1.5,
        gamma=1.25,
        reml=False,
        weights=weights,
        n_effective_total=8.0,
    )
    reml = profiled_gaussian_reml_variance(
        10.0,
        2.0,
        4.0,
        1.5,
        gamma=1.25,
        reml=True,
        weights=weights,
        n_effective_total=8.0,
    )

    assert ml == pytest.approx(12.0 / 6.0)
    assert reml == pytest.approx(12.0 / (6.0 - 1.25 * 1.5))


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


def _build_runtime_term(data: pd.DataFrame, formula: str):
    parsed = parse_gam_formula(formula)
    extracted = extract_formula_terms(parsed)
    built = build_formula_model(extracted, data=data, y=np.zeros(len(data)))
    predictor = built.predictor_specs[0]
    assert len(predictor.terms) == 1
    term = instantiate_term(predictor.terms[0])
    term.fit(built.X, built.feature_names)
    return term


def test_gamma_newton_branch_exposes_distinct_working_and_fisher_weights():
    """
    Regression coverage verifying that gamma newton branch exposes distinct working and
    fisher weights.
    """
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(240, 2))
    eta = 0.3 + 0.7 * np.sin(X[:, 0]) - 0.2 * X[:, 1]
    mu = np.exp(eta)
    shape = 3.0
    y = rng.gamma(shape=shape, scale=mu / shape)

    gam = GAM(
        k=8,
        family={"name": "gamma", "link": "log"},
        optimize_smoothing=False,
        smoothing_method="fixed",
    )
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
    """
    Regression coverage verifying that PIRLS step halving exhaustion returns failure
    without accepting bad step.
    """
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
    """
    Regression coverage verifying that binomial PIRLS uses stacked QR when system is ill
    conditioned.
    """
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


def test_get_r_pqr_serial_handles_wide_qr_storage():
    """Regression coverage verifying that get r pqr serial handles wide QR storage."""
    rng = np.random.default_rng(123)
    X = rng.normal(size=(4, 7))

    qr_a, _tau, _pivot, _ = _dgeqp3_economic_r(X)
    got = _get_r_pqr_serial(qr_a, rr=min(X.shape), ncol=X.shape[1])

    want = np.triu(np.asarray(qr_a[: min(X.shape), :], dtype=np.float64))
    np.testing.assert_allclose(got, want, atol=1e-12, rtol=1e-12)


def test_stacked_qr_covariance_root_scatter_matches_covariance():
    """
    Regression coverage verifying that stacked QR covariance root scatter matches
    covariance.
    """
    upper_r_final = np.array([[3.0, 1.0], [0.0, 2.0]], dtype=np.float64)
    pivot1 = np.array([1, 0], dtype=np.int64)
    kept = [0, 2]

    root, cov = stacked_qr_covariance_from_factor(
        upper_r_final,
        pivot1=pivot1,
        kept_original_indices=kept,
        q_total=4,
    )

    np.testing.assert_allclose(cov, root @ root.T, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12, rtol=1e-12)


def test_gaussian_stacked_qr_rank_deficient_signed_weights_match_reduced_problem():
    """
    Regression coverage verifying that gaussian stacked QR rank deficient signed weights
    match reduced problem.
    """
    X = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    w = np.array([1.0, -0.25, 1.0], dtype=np.float64)
    lam = 1e-12
    P = lam * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)

    out = solve_gaussian_penalized_ls_stacked_qr(
        X,
        y,
        w,
        P,
        fit_intercept=False,
        n_coef=X.shape[1],
    )

    t_hat = float(np.dot(w, y) / (np.sum(w) + lam))
    eta_expected = np.full(X.shape[0], t_hat, dtype=np.float64)

    assert int(out["penalized_system_rank"]) == 1
    np.testing.assert_allclose(
        np.asarray(out["eta"], dtype=np.float64),
        eta_expected,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        X @ np.asarray(out["coef_full"], dtype=np.float64),
        eta_expected,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        float(out["penalty_quadratic"]),
        lam * t_hat * t_hat,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(out["A_inv"], dtype=np.float64),
        np.asarray(out["covariance_root"], dtype=np.float64)
        @ np.asarray(out["covariance_root"], dtype=np.float64).T,
        atol=1e-12,
        rtol=1e-12,
    )


def test_stacked_qr_null_space_gauge_preserves_fit_and_minimizes_penalty():
    """
    Regression coverage for the explicit mgcv boundary tie-break used by exact
    Gaussian stacked-QR fits: fitted values are invariant, while the coefficient
    representative is selected by minimum penalty within null(X).
    """
    X = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=np.float64,
    )
    y = np.array([1.5, 3.0, 4.5], dtype=np.float64)
    w = np.ones(3, dtype=np.float64)
    P = np.diag([1e-12, 1.0]).astype(np.float64)

    ungauged = solve_gaussian_penalized_ls_stacked_qr(
        X,
        y,
        w,
        P,
        fit_intercept=False,
        n_coef=X.shape[1],
        near_singular_null_pin=False,
    )
    gauged = solve_gaussian_penalized_ls_stacked_qr(
        X,
        y,
        w,
        P,
        fit_intercept=False,
        n_coef=X.shape[1],
        near_singular_null_pin=True,
    )

    beta_g = np.asarray(gauged["coef_full"], dtype=np.float64)
    null_dir = np.array([1.0, -1.0], dtype=np.float64) / np.sqrt(2.0)

    np.testing.assert_allclose(
        X @ beta_g,
        X @ np.asarray(ungauged["coef_full"], dtype=np.float64),
        atol=1e-10,
        rtol=1e-10,
    )
    assert abs(float(null_dir @ (P @ beta_g))) < 1e-10
    assert (
        float(beta_g @ (P @ beta_g))
        <= float(
            np.asarray(ungauged["coef_full"], dtype=np.float64)
            @ (P @ np.asarray(ungauged["coef_full"], dtype=np.float64))
        )
        + 1e-12
    )


def test_irls_zero_penalty_rank_deficient_design_uses_qr_drop():
    """
    Regression coverage for mgcv::pls_fit1 parity when the penalty has zero rows:
    rank-deficient unpenalized working systems are rank-dropped by QR, not solved
    by dense Cholesky normal equations.
    """
    X = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ],
        dtype=np.float64,
    )
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    out = irls_core(
        X,
        y,
        GaussianIdentityFamily(),
        np.zeros((2, 2), dtype=np.float64),
        fit_intercept=False,
        max_iter=1,
    )

    assert int(out["penalized_system_rank"]) == 1
    assert np.asarray(out["dropped_column_indices"], dtype=np.int64).size == 1
    np.testing.assert_allclose(out["eta"], y, atol=1e-12, rtol=1e-12)


def test_signed_weight_penalized_qr_state_matches_stacked_gaussian_solve():
    """
    Regression coverage verifying that signed weight penalized QR state matches stacked
    gaussian solve.
    """
    X = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    w = np.array([1.0, -0.25, 1.0], dtype=np.float64)
    lam = 1e-12
    P = lam * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)

    out = solve_gaussian_penalized_ls_stacked_qr(
        X,
        y,
        w,
        P,
        fit_intercept=False,
        n_coef=X.shape[1],
    )
    E, Es = penalty_sqrt_rows(P)
    qr_state = build_penalized_qr_state_nonnegative(
        X,
        y,
        w,
        penalty_sqrt_E=E,
        penalty_rank_Es=Es,
        rS=E.T,
        rank_tol=STACKED_QR_RANK_TOLERANCE,
        reml=True,
    )

    state_root_full = _scatter_pivoted_rank_matrix_to_full(
        np.asarray(qr_state.P, dtype=np.float64),
        kept_original_indices=qr_state.kept_original_indices,
        pivot1=qr_state.pivot1,
        q_total=X.shape[1],
    )

    assert int(qr_state.rank) == int(out["penalized_system_rank"])
    np.testing.assert_allclose(
        np.asarray(qr_state.beta_full, dtype=np.float64),
        np.asarray(out["coef_full"], dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        float(qr_state.ldet_XWX_plus_S),
        float(out["log_det_XtWX_plus_penalty"]),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        state_root_full,
        np.asarray(out["covariance_root"], dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_gdi_pk_setup_and_ift1_match_signed_weight_inverse_root(monkeypatch):
    """
    Regression coverage verifying that GDI pk setup and IFT1 match signed weight inverse
    root.
    """
    pirls_deriv_module = importlib.import_module(
        "nampy.gam.smoothing_selection.criteria.pirls.derivatives"
    )

    X = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    w = np.array([1.0, -0.25, 1.0], dtype=np.float64)
    lam = 1e-12
    sp = np.array([lam], dtype=np.float64)
    P = lam * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)

    out = solve_gaussian_penalized_ls_stacked_qr(
        X,
        y,
        w,
        P,
        fit_intercept=False,
        n_coef=X.shape[1],
    )
    E, Es = penalty_sqrt_rows(P)
    canonical = SimpleNamespace(
        T=np.eye(X.shape[1], dtype=np.float64),
        St=P.copy(),
        Sr=E.copy(),
        Eb=Es.copy(),
        Mp=0,
        rp={"rS": [E.T.copy()]},
    )
    monkeypatch.setattr(
        pirls_deriv_module,
        "build_penalty_reparameterization_state",
        lambda *args, **kwargs: canonical,
    )

    model = SimpleNamespace(
        family=GaussianIdentityFamily(),
        compiled_model_=SimpleNamespace(n_smoothing_params=1),
    )
    sol = {
        "X": X,
        "coef_full": np.asarray(out["coef_full"], dtype=np.float64),
        "working_weights": w,
        "eta": np.asarray(out["eta"], dtype=np.float64),
        "mu": np.asarray(out["eta"], dtype=np.float64),
    }

    setup = pirls_deriv_module._gdi_pk_setup(model, sol, sp, deriv=2)
    current = setup.current
    pk_state = setup.pk
    ift = pirls_deriv_module._gdi1_ift1_state(model, y, sol, sp, current, pk_state)

    np.testing.assert_allclose(
        np.asarray(current.W, dtype=np.float64),
        w,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(current.A_inv, dtype=np.float64),
        np.asarray(pk_state.P, dtype=np.float64)
        @ np.asarray(pk_state.P, dtype=np.float64).T,
        atol=1e-12,
        rtol=1e-12,
    )

    expected_dbeta = -np.asarray(current.A_inv, dtype=np.float64) @ (
        np.asarray(ift.P_derivs[0], dtype=np.float64) @ np.asarray(current.beta)
    )
    expected_d2beta = (
        -2.0
        * np.asarray(current.A_inv, dtype=np.float64)
        @ (
            np.asarray(ift.P_derivs[0], dtype=np.float64)
            @ np.asarray(ift.dbeta[0], dtype=np.float64)
        )
    )
    expected_d2beta = expected_d2beta + np.asarray(ift.dbeta[0], dtype=np.float64)

    np.testing.assert_allclose(
        np.asarray(ift.dbeta[0], dtype=np.float64),
        expected_dbeta,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(ift.d2beta_mat[0][0], dtype=np.float64),
        expected_d2beta,
        atol=1e-12,
        rtol=1e-12,
    )


def test_gaussian_fit3_gdi_beta_full_matches_signed_weight_stacked_qr(monkeypatch):
    """
    Regression coverage verifying that gaussian fit3 GDI beta full matches signed weight
    stacked QR.
    """
    gaussian_exact_module = importlib.import_module(
        "nampy.gam.fit.solvers.gaussian_exact"
    )
    reparam_module = importlib.import_module("nampy.gam.smoothing_selection.reparam")

    X = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    w = np.array([1.0, -0.25, 1.0], dtype=np.float64)
    lam = 1e-12
    sp = np.array([lam], dtype=np.float64)
    P = lam * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    E, Es = penalty_sqrt_rows(P)

    canonical = SimpleNamespace(
        T=np.eye(X.shape[1], dtype=np.float64),
        Sr=E.copy(),
        Eb=Es.copy(),
        Mp=0,
        rp={"rS": [E.T.copy()]},
    )
    monkeypatch.setattr(
        reparam_module,
        "build_penalty_reparameterization_state",
        lambda *args, **kwargs: canonical,
    )

    model = SimpleNamespace(compiled_model_=SimpleNamespace(n_smoothing_params=1))
    coef_full, eta_fit, _rank_root = gaussian_exact_module._gaussian_fit3_gdi_beta_full(
        model,
        X,
        sp,
        y,
        w,
    )
    out = solve_gaussian_penalized_ls_stacked_qr(
        X,
        y,
        w,
        P,
        fit_intercept=False,
        n_coef=X.shape[1],
    )

    np.testing.assert_allclose(
        np.asarray(coef_full, dtype=np.float64),
        np.asarray(out["coef_full"], dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(eta_fit, dtype=np.float64),
        np.asarray(out["eta"], dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_gaussian_exact_score_refuses_fs_and_sz_surfaces():
    """
    Regression coverage verifying that gaussian exact score refuses fs and sz surfaces.
    """
    data = _make_gaussian_data(seed=17, n=60).rename(columns={"x0": "x"})

    fs = GAM(
        family="gaussian",
        formula='y ~ s(f, x, bs="fs", k=6)',
        optimize_smoothing=False,
    )
    fs_data = pd.DataFrame(
        {
            "y": data["y"],
            "x": data["x"],
            "f": np.asarray(["a", "b", "c"] * 20, dtype=object),
        }
    )
    fs.fit(data=fs_data)
    fs_log_sp = np.log(
        np.maximum(np.asarray(fs.smoothing_params, dtype=np.float64), 1e-12)
    )
    assert criterion_ml_reml_exact(fs, fs.y_, fs_log_sp, "REML") == np.inf

    sz = GAM(
        family="gaussian",
        formula='y ~ s(f1, f2, x, bs="sz", k=6)',
        optimize_smoothing=False,
    )
    sz_data = pd.DataFrame(
        {
            "y": data["y"].iloc[:18].to_numpy(),
            "x": data["x"].iloc[:18].to_numpy(),
            "f1": np.asarray(["a", "b", "c"] * 6, dtype=object),
            "f2": np.asarray((["u"] * 3 + ["v"] * 3 + ["w"] * 3) * 2, dtype=object),
        }
    )
    sz.fit(data=sz_data)
    sz_log_sp = np.log(
        np.maximum(np.asarray(sz.smoothing_params, dtype=np.float64), 1e-12)
    )
    assert criterion_ml_reml_exact(sz, sz.y_, sz_log_sp, "REML") == np.inf


def test_assign_fit_solution_transforms_gaussian_unconditional_covariance(
    monkeypatch,
):
    """
    Regression coverage verifying that assign fit solution transforms gaussian
    unconditional covariance.
    """
    uncond_module = importlib.import_module(
        "nampy.gam.fit.postprocess.unconditional_covariance"
    )

    P = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cov_bayes = np.array(
        [
            [2.0, 0.2, 0.1],
            [0.2, 1.5, 0.3],
            [0.1, 0.3, 1.2],
        ],
        dtype=np.float64,
    )
    cov_freq = np.array(
        [
            [1.8, 0.1, 0.0],
            [0.1, 1.3, 0.2],
            [0.0, 0.2, 1.0],
        ],
        dtype=np.float64,
    )
    cov_unconditional = np.array(
        [
            [2.4, 0.3, 0.2],
            [0.3, 1.9, 0.4],
            [0.2, 0.4, 1.6],
        ],
        dtype=np.float64,
    )

    monkeypatch.setattr(
        uncond_module,
        "_pirls_exact_unconditional_postfit",
        lambda *args, **kwargs: (None, None, "fit"),
    )
    monkeypatch.setattr(
        uncond_module,
        "_gaussian_exact_unconditional_postfit",
        lambda *args, **kwargs: (cov_unconditional.copy(), None, "fit"),
    )

    fit_state = FitState(
        X=np.eye(3, dtype=np.float64),
        A=np.eye(3, dtype=np.float64),
        A_inv=np.eye(3, dtype=np.float64),
        XtWX=np.eye(3, dtype=np.float64),
        P=np.zeros((3, 3), dtype=np.float64),
        working_weights=np.ones(3, dtype=np.float64),
        fisher_weights=np.ones(3, dtype=np.float64),
        scale=1.0,
    )
    fit_result = FitResult(
        coef_full=np.array([1.0, -0.5, 0.25], dtype=np.float64),
        intercept=0.0,
        beta=np.array([1.0, -0.5, 0.25], dtype=np.float64),
        eta=np.zeros(3, dtype=np.float64),
        mu=np.zeros(3, dtype=np.float64),
        rss=0.0,
        deviance=0.0,
        edf=0.0,
        trace_H=0.0,
        scale=1.0,
        cov_bayes=cov_bayes,
        cov_freq=cov_freq,
        cov_unconditional=None,
        H_coef=np.zeros((3, 3), dtype=np.float64),
    )
    sol = FitCoreSolution(
        fit_result=fit_result,
        fit_state=fit_state,
        penalized_system=fit_state.to_penalized_system(),
    )

    model = SimpleNamespace(
        family=GaussianIdentityFamily(),
        fit_intercept=False,
        n_samples_=3,
        compiled_model_=SimpleNamespace(
            metadata={"fit_to_prediction_parameterization_map": P},
            compiled_terms=(),
            compiled_penalties=(),
            predictors=(),
            predictor_full_slices=(),
            n_coef=3,
            n_smoothing_params=1,
        ),
        fit_core_solution_=None,
        gam_result_=None,
    )

    assign_fit_solution(model, sol)

    got = model.fit_core_solution_.fit_result
    np.testing.assert_allclose(
        got.cov_bayes, P @ cov_bayes @ P.T, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(got.cov_freq, P @ cov_freq @ P.T, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        got.cov_unconditional,
        P @ cov_unconditional @ P.T,
        atol=1e-12,
        rtol=1e-12,
    )


def test_assign_fit_solution_transforms_pirls_unconditional_covariance_and_edf2(
    monkeypatch,
):
    """
    Regression coverage verifying that assign fit solution transforms PIRLS
    unconditional covariance and edf2.
    """
    uncond_module = importlib.import_module(
        "nampy.gam.fit.postprocess.unconditional_covariance"
    )

    P = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    cov_bayes = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.7],
        ],
        dtype=np.float64,
    )
    cov_unconditional = np.array(
        [
            [1.2, 0.2],
            [0.2, 0.8],
        ],
        dtype=np.float64,
    )
    edf2 = np.array([0.4, 0.3], dtype=np.float64)

    monkeypatch.setattr(
        uncond_module,
        "_pirls_exact_unconditional_postfit",
        lambda *args, **kwargs: (cov_unconditional.copy(), edf2.copy(), "fit"),
    )
    monkeypatch.setattr(
        uncond_module,
        "_gaussian_exact_unconditional_postfit",
        lambda *args, **kwargs: (None, None, "fit"),
    )

    fit_state = FitState(
        X=np.eye(2, dtype=np.float64),
        A=np.eye(2, dtype=np.float64),
        A_inv=np.eye(2, dtype=np.float64),
        XtWX=np.eye(2, dtype=np.float64),
        P=np.zeros((2, 2), dtype=np.float64),
        working_weights=np.ones(2, dtype=np.float64),
        fisher_weights=np.ones(2, dtype=np.float64),
        scale=1.0,
    )
    fit_result = FitResult(
        coef_full=np.array([0.2, -0.1], dtype=np.float64),
        intercept=0.0,
        beta=np.array([0.2, -0.1], dtype=np.float64),
        eta=np.zeros(2, dtype=np.float64),
        mu=np.full(2, 0.5, dtype=np.float64),
        rss=None,
        deviance=0.0,
        edf=0.0,
        trace_H=0.0,
        scale=1.0,
        cov_bayes=cov_bayes,
        cov_freq=cov_bayes.copy(),
        cov_unconditional=None,
        H_coef=np.eye(2, dtype=np.float64),
    )
    sol = FitCoreSolution(
        fit_result=fit_result,
        fit_state=fit_state,
        penalized_system=fit_state.to_penalized_system(),
    )

    model = SimpleNamespace(
        family=BinomialLogitFamily(),
        fit_intercept=False,
        n_samples_=2,
        compiled_model_=SimpleNamespace(
            metadata={"fit_to_prediction_parameterization_map": P},
            compiled_terms=(),
            compiled_penalties=(),
            predictors=(),
            predictor_full_slices=(),
            n_coef=2,
            n_smoothing_params=1,
        ),
        fit_core_solution_=None,
        gam_result_=None,
    )

    assign_fit_solution(model, sol)

    got = model.fit_core_solution_.fit_result
    np.testing.assert_allclose(
        got.cov_unconditional,
        P @ cov_unconditional @ P.T,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(got.edf2, edf2, atol=1e-12, rtol=1e-12)
    assert got.cov_unconditional_space == "prediction"


def test_pirls_unconditional_postfit_uses_edge_correct_vc_but_fitted_edf2(
    monkeypatch,
):
    """
    Regression coverage verifying that PIRLS unconditional postfit uses edge correct vc
    but fitted edf2.
    """
    uncond_module = importlib.import_module(
        "nampy.gam.fit.postprocess.unconditional_covariance"
    )
    capabilities_module = importlib.import_module("nampy.gam.fit.capabilities")
    criteria_dispatch_module = importlib.import_module(
        "nampy.gam.smoothing_selection.criteria.dispatch"
    )
    pirls_deriv_module = importlib.import_module(
        "nampy.gam.smoothing_selection.criteria.pirls.derivatives"
    )
    reparam_module = importlib.import_module("nampy.gam.smoothing_selection.reparam")
    newton_solver_module = importlib.import_module(
        "nampy.gam.fit.solvers.general_family.newton"
    )

    captured_rho = []

    def _capture_vb_corr_root(
        X_root,
        *,
        L,
        lsp0,
        S_blocks,
        off,
        rho,
        Vr,
        scale_est=False,
        **kwargs,
    ):
        del X_root, L, lsp0, S_blocks, off, Vr, scale_est, kwargs
        rho = np.asarray(rho, dtype=np.float64).copy()
        captured_rho.append(rho)
        if np.allclose(rho, np.array([np.log(0.5)], dtype=np.float64)):
            return np.array([[0.02]], dtype=np.float64)
        return np.array([[0.03]], dtype=np.float64)

    monkeypatch.setattr(
        capabilities_module,
        "can_use_simple_ml_reml_structure",
        lambda model: True,
    )
    monkeypatch.setattr(
        criteria_dispatch_module,
        "criterion_hessian",
        lambda *args, **kwargs: np.array([[4.0]], dtype=np.float64),
    )
    monkeypatch.setattr(
        pirls_deriv_module,
        "_gdi1_kernel",
        lambda *args, **kwargs: SimpleNamespace(
            current=SimpleNamespace(
                pivot1=np.array([0], dtype=np.int64),
                dropped_column_indices=np.array([], dtype=np.int64),
                canonical=SimpleNamespace(T=np.array([[1.0]], dtype=np.float64)),
            ),
            ift=SimpleNamespace(dbeta=[np.array([0.2], dtype=np.float64)]),
        ),
    )
    monkeypatch.setattr(
        reparam_module,
        "build_estimate_gam_setup_state",
        lambda *args, **kwargs: SimpleNamespace(
            S=[np.array([[0.0]], dtype=np.float64)],
            off=np.array([1], dtype=np.int64),
            L=None,
            lsp0=np.array([0.0], dtype=np.float64),
        ),
    )
    monkeypatch.setattr(newton_solver_module, "_vb_corr_root", _capture_vb_corr_root)

    model = SimpleNamespace(
        family=BinomialLogitFamily(),
        _optim_method="reml",
        _optim_result=SimpleNamespace(
            outer_info={
                "hess1": np.array([[9.0]], dtype=np.float64),
                "db_drho1": np.array([[0.4]], dtype=np.float64),
                "lsp1": np.array([-1.5], dtype=np.float64),
            }
        ),
        smoothing_fixed_mask_=None,
        smoothing_params=np.array([0.5], dtype=np.float64),
        y_=np.array([0.0], dtype=np.float64),
        compiled_model_=SimpleNamespace(
            n_smoothing_params=1,
            metadata={},
            compiled_terms=(),
            compiled_penalties=(),
            predictors=(),
            predictor_full_slices=(),
            n_coef=1,
        ),
    )
    fit_state = FitState(
        X=np.array([[1.0]], dtype=np.float64),
        A=np.array([[1.0]], dtype=np.float64),
        A_inv=np.array([[1.0]], dtype=np.float64),
        XtWX=np.array([[1.0]], dtype=np.float64),
        P=np.zeros((1, 1), dtype=np.float64),
        fisher_weights=np.array([1.0], dtype=np.float64),
        working_weights=np.array([1.0], dtype=np.float64),
        scale=1.0,
    )
    fit_result = FitResult(
        coef_full=np.array([0.0], dtype=np.float64),
        intercept=0.0,
        beta=np.array([0.0], dtype=np.float64),
        eta=np.array([0.0], dtype=np.float64),
        mu=np.array([0.5], dtype=np.float64),
        rss=None,
        deviance=0.0,
        edf=0.0,
        trace_H=0.5,
        scale=1.0,
        cov_bayes=np.array([[0.1]], dtype=np.float64),
        cov_freq=np.array([[0.1]], dtype=np.float64),
        cov_unconditional=None,
        H_coef=np.array([[0.5]], dtype=np.float64),
    )

    Vc, edf2, space = uncond_module._pirls_exact_unconditional_postfit(
        model,
        SimpleNamespace(),
        fit_result,
        fit_state,
    )

    np.testing.assert_allclose(
        captured_rho[0],
        np.array([np.log(0.5)], dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        captured_rho[1],
        np.array([-1.5], dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        Vc,
        np.array([[0.1 + (0.4**2) / 9.0 + 0.03]], dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        edf2,
        np.array([0.1 + (0.2**2) / 4.0 + 0.02], dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    assert space == "fit"


def test_gaussian_unconditional_postfit_augments_link_matrix_for_joint_scale(
    monkeypatch,
):
    """
    Regression coverage verifying that gaussian unconditional postfit augments link
    matrix for joint scale.
    """
    uncond_module = importlib.import_module(
        "nampy.gam.fit.postprocess.unconditional_covariance"
    )
    backends_module = importlib.import_module("nampy.gam.fit.backends")
    gaussian_dyn_module = importlib.import_module(
        "nampy.gam.smoothing_selection.criteria.gaussian_dyn"
    )
    pirls_deriv_module = importlib.import_module(
        "nampy.gam.smoothing_selection.criteria.pirls.derivatives"
    )
    reparam_module = importlib.import_module("nampy.gam.smoothing_selection.reparam")
    newton_solver_module = importlib.import_module(
        "nampy.gam.fit.solvers.general_family.newton"
    )

    captured = {}

    def _capture_vb_corr_root(
        X_root,
        *,
        L,
        lsp0,
        S_blocks,
        off,
        rho,
        Vr,
        scale_est=False,
    ):
        captured["L"] = None if L is None else np.asarray(L, dtype=np.float64).copy()
        captured["lsp0"] = np.asarray(lsp0, dtype=np.float64).copy()
        captured["rho"] = np.asarray(rho, dtype=np.float64).copy()
        captured["scale_est"] = bool(scale_est)
        return np.zeros((1, 1), dtype=np.float64)

    monkeypatch.setattr(
        gaussian_dyn_module,
        "criterion_hessian_ml_reml_gaussian_dynamic_joint",
        lambda *args, **kwargs: np.array([[4.0, 0.0], [0.0, 9.0]], dtype=np.float64),
    )
    monkeypatch.setattr(
        backends_module,
        "solve_gaussian_given_smoothing",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        pirls_deriv_module,
        "_gdi1_kernel",
        lambda *args, **kwargs: SimpleNamespace(
            current=SimpleNamespace(
                R=np.array([[1.0]], dtype=np.float64),
                pivot1=np.array([0], dtype=np.int64),
                dropped_column_indices=np.array([], dtype=np.int64),
                canonical=SimpleNamespace(T=np.array([[1.0]], dtype=np.float64)),
            ),
            ift=SimpleNamespace(dbeta=[np.array([0.0], dtype=np.float64)]),
        ),
    )
    monkeypatch.setattr(
        reparam_module,
        "build_estimate_gam_setup_state",
        lambda *args, **kwargs: SimpleNamespace(
            S=[np.array([[2.0]], dtype=np.float64)],
            off=np.array([1], dtype=np.int64),
            L=np.array([[3.0]], dtype=np.float64),
            lsp0=np.array([0.25], dtype=np.float64),
        ),
    )
    monkeypatch.setattr(newton_solver_module, "_vb_corr_root", _capture_vb_corr_root)

    model = SimpleNamespace(
        family=GaussianIdentityFamily(),
        _optim_method="reml",
        _optim_result=SimpleNamespace(joint_log_sigma2=float(np.log(0.5))),
        _gaussian_reml_sigma2_opt_=0.5,
        smoothing_fixed_mask_=None,
        smoothing_params=np.array([0.75], dtype=np.float64),
        score_gamma=1.0,
        y_=np.array([0.0], dtype=np.float64),
        compiled_model_=SimpleNamespace(
            n_smoothing_params=1,
            metadata={},
            compiled_terms=(),
            compiled_penalties=(),
            predictors=(),
            predictor_full_slices=(),
            n_coef=1,
        ),
    )
    fit_state = FitState(
        X=np.array([[1.0]], dtype=np.float64),
        A=np.array([[1.0]], dtype=np.float64),
        A_inv=np.array([[1.0]], dtype=np.float64),
        XtWX=np.array([[1.0]], dtype=np.float64),
        P=np.zeros((1, 1), dtype=np.float64),
        fisher_weights=np.array([1.0], dtype=np.float64),
        working_weights=np.array([1.0], dtype=np.float64),
        scale=0.5,
    )
    fit_result = FitResult(
        coef_full=np.array([0.0], dtype=np.float64),
        intercept=0.0,
        beta=np.array([0.0], dtype=np.float64),
        eta=np.array([0.0], dtype=np.float64),
        mu=np.array([0.0], dtype=np.float64),
        rss=0.0,
        deviance=0.0,
        edf=0.0,
        trace_H=1.0,
        scale=0.5,
        cov_bayes=np.array([[0.5]], dtype=np.float64),
        cov_freq=np.array([[0.5]], dtype=np.float64),
        cov_unconditional=None,
        H_coef=np.array([[0.25]], dtype=np.float64),
    )

    uncond_module._gaussian_exact_unconditional_postfit(model, fit_result, fit_state)

    assert captured["scale_est"] is True
    np.testing.assert_allclose(
        captured["L"],
        np.array([[3.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        captured["lsp0"],
        np.array([0.25, 0.0], dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        captured["rho"],
        np.array([np.log(0.75), np.log(0.5)], dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_outer_newton_result_sets_stable_metadata():
    """Regression coverage verifying that outer newton result sets stable metadata."""

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


def test_mgcv_outer_newton_steepest_descent_fallback_uses_negative_gradient():
    """
    Regression coverage verifying that mgcv outer newton steepest descent fallback uses
    the negative-gradient direction required by the validated mgcv trace.
    """
    target = np.array([-0.09882118, -0.09882118], dtype=np.float64)

    class _Obj:
        def __init__(self):
            self.n_fun = 0
            self.n_jac = 0
            self.n_hess = 0

    objective = _Obj()

    def _eval_at(
        x_eval,
        *,
        start_coef,
        start_eta,
        start_mu,
        need_grad,
        need_hess,
        commit_start=False,
    ):
        x_eval = np.asarray(x_eval, dtype=np.float64).ravel()
        if np.linalg.norm(x_eval - target) < 1e-6:
            score = 9.8
            grad = np.zeros(2, dtype=np.float64)
            hess = np.eye(2, dtype=np.float64)
        elif np.all(x_eval > 0.0):
            score = 20.0 + float(np.dot(x_eval, x_eval))
            grad = np.array([1.0, 1.0], dtype=np.float64)
            hess = np.array([[2.0, 0.0], [0.0, -1.0]], dtype=np.float64)
        elif np.linalg.norm(x_eval) < 1e-12:
            score = 10.0
            grad = np.array([1.0, 1.0], dtype=np.float64)
            hess = np.array([[2.0, 0.0], [0.0, -1.0]], dtype=np.float64)
        else:
            score = 20.0 + float(np.dot(x_eval + 0.3, x_eval + 0.3))
            grad = np.array([1.0, 1.0], dtype=np.float64)
            hess = np.array([[2.0, 0.0], [0.0, -1.0]], dtype=np.float64)

        objective.n_fun += 1
        if need_grad:
            objective.n_jac += 1
        if need_hess:
            objective.n_hess += 1
        return (
            float(score),
            None if not need_grad else grad,
            None if not need_hess else hess,
            np.full(x_eval.shape, np.nan, dtype=np.float64),
            None,
            None,
            None,
            1.0,
        )

    out = _optimize_outer_newton_strict(
        objective=objective,
        x0=np.zeros(2, dtype=np.float64),
        bounds=[(-2.0, 2.0), (-2.0, 2.0)],
        eval_at=_eval_at,
        max_iter=5,
        max_half=8,
    )

    np.testing.assert_allclose(out.x, target, atol=1e-6, rtol=0.0)
    assert bool(out.optim_trace[0]["rank_info"]["used_steepest_descent"])


def test_mgcv_outer_newton_step_failure_does_not_report_success():
    """
    Regression coverage verifying that mgcv outer newton preserves step-failure
    semantics even when the trial step is tiny.
    """

    class _Obj:
        def __init__(self):
            self.n_fun = 0
            self.n_jac = 0
            self.n_hess = 0

    objective = _Obj()

    def _eval_at(
        x_eval,
        *,
        start_coef,
        start_eta,
        start_mu,
        need_grad,
        need_hess,
        commit_start=False,
    ):
        del start_coef, start_eta, start_mu, commit_start
        x_eval = np.asarray(x_eval, dtype=np.float64).ravel()
        score = 10.0 if np.linalg.norm(x_eval) < 1e-12 else 12.0
        grad = np.array([1.0], dtype=np.float64)
        hess = np.array([[-1.0]], dtype=np.float64)
        objective.n_fun += 1
        if need_grad:
            objective.n_jac += 1
        if need_hess:
            objective.n_hess += 1
        return (
            float(score),
            None if not need_grad else grad,
            None if not need_hess else hess,
            np.full(grad.shape, np.nan, dtype=np.float64),
            None,
            None,
            None,
            1.0,
        )

    out = _optimize_outer_newton_strict(
        objective=objective,
        x0=np.zeros(1, dtype=np.float64),
        bounds=[(-1e-9, 1e-9)],
        eval_at=_eval_at,
        max_iter=5,
        max_half=3,
        step_tol=1.0,
    )

    assert bool(out.success) is False
    assert str(out.message) == "step failed"


def test_tp_constructor_uses_covariate_locations_when_supplied_knots_are_too_few():
    """
    Regression coverage verifying that tp setup falls back to covariate locations
    instead of silently shrinking `k` when supplied knots are fewer than `k`.
    """
    data = pd.DataFrame(
        {
            "y": np.sin(np.linspace(0.0, 1.0, 24, dtype=np.float64)),
            "x": np.linspace(0.0, 1.0, 24, dtype=np.float64),
        }
    )

    out = construct_tprs_basis(
        data[["x"]].to_numpy(dtype=np.float64),
        k=6,
        penalty_order=2,
        setup_locations=np.linspace(0.0, 1.0, 4, dtype=np.float64).reshape(-1, 1),
    )

    assert out["X_raw"].shape[1] == 6


def test_prediction_parameterization_respects_public_space_covariance_tags():
    """
    Regression coverage verifying that prediction parameterization respects public space
    covariance tags.
    """
    state_module = importlib.import_module("nampy.gam.fit.state")

    P = np.array(
        [
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cov_bayes = np.array(
        [
            [2.0, 0.3],
            [0.3, 1.5],
        ],
        dtype=np.float64,
    )
    cov_unconditional_public = np.array(
        [
            [5.0, 0.4],
            [0.4, 1.8],
        ],
        dtype=np.float64,
    )

    fit_result = FitResult(
        coef_full=np.array([1.0, -0.5], dtype=np.float64),
        intercept=0.0,
        beta=np.array([1.0, -0.5], dtype=np.float64),
        eta=np.zeros(2, dtype=np.float64),
        mu=np.zeros(2, dtype=np.float64),
        rss=0.0,
        deviance=0.0,
        edf=0.0,
        trace_H=0.0,
        scale=1.0,
        cov_bayes=cov_bayes,
        cov_freq=cov_bayes.copy(),
        cov_unconditional=cov_unconditional_public,
        H_coef=np.eye(2, dtype=np.float64),
        cov_unconditional_space="prediction",
    )
    model = SimpleNamespace(
        family=GaussianIdentityFamily(),
        compiled_model_=SimpleNamespace(
            metadata={"fit_to_prediction_parameterization_map": P},
            compiled_terms=(),
        ),
    )

    out = state_module._apply_prediction_parameterization_to_fit_result(
        model,
        fit_result,
        None,
    )

    np.testing.assert_allclose(
        out.cov_bayes,
        P @ cov_bayes @ P.T,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        out.cov_unconditional,
        cov_unconditional_public,
        atol=1e-12,
        rtol=1e-12,
    )


def test_disjoint_multi_penalty_term_is_accepted_and_reparameterized():
    """
    Regression coverage verifying that disjoint multi penalty term is accepted and
    reparameterized.
    """

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
    """
    Regression coverage verifying that overlapping null space penalties on one term are
    accepted.
    """

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
    """
    Regression coverage verifying that dynamic reparam design depends on current sp.
    """

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
    """
    Regression coverage verifying that canonical gam reparam state matches dynamic
    design.
    """
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
    canonical = build_penalty_reparameterization_state(model, x_full, sp, deriv=1)
    dynamic = dynamic_reparam_design(model, x_full, sp)

    assert canonical.U1.shape == (5, 5)
    assert canonical.T.shape == (5, 5)
    assert canonical.St.shape == (5, 5)
    assert canonical.Sr.shape[1] == 5
    assert canonical.Mp == 1
    np.testing.assert_allclose(canonical.X_fix, dynamic.X_fix, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(canonical.Z_rand, dynamic.Z_rand, rtol=0.0, atol=1e-12)


def test_gam_fit3_state_groups_shared_id_from_exact_estimate_setup():
    """
    Regression coverage verifying that gam fit3 state groups shared id from exact
    estimate setup.
    """
    data = _make_gaussian_data(seed=777, n=120)
    gam = GAM(
        family="gaussian",
        formula='y ~ s(x0, bs="cr", k=8, id="g") + s(x1, bs="cr", k=8, id="g")',
        optimize_smoothing=False,
        smoothing_method="REML",
    )
    gam.fit(data=data)

    estimate = build_estimate_gam_setup_state(gam)
    state = build_penalty_reparameterization_state(
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
def test_mixed_list_xt_payload_survives_fit_for_tp_and_fs():
    """
    Regression coverage verifying that mixed list xt payload survives fit for tp and
    fs terms.
    """
    data = pd.DataFrame(
        {
            "y": np.linspace(0.1, 4.0, 40, dtype=np.float64),
            "x": np.linspace(0.0, 1.0, 40, dtype=np.float64),
            "f": np.asarray(["a", "b"] * 20, dtype=object),
        }
    )
    cases = [
        (
            'y ~ s(x, bs="tp", k=5, xt=list(1, seed=2))',
            {0: 1, "seed": 2},
        ),
        (
            'y ~ s(f, x, bs="fs", k=5, xt=list(1, bs="ps", m=2))',
            {0: 1, "bs": "ps", "m": 2},
        ),
    ]

    for formula, expected_xt in cases:
        gam = GAM(
            family="gaussian",
            formula=formula,
            optimize_smoothing=False,
            smoothing_method="fixed",
        )
        gam.fit(data=data)

        term = gam.compiled_model_.compiled_terms[0]
        term_spec = term.metadata["term_spec"]
        assert term_spec["basis_options"]["xt"] == expected_xt


def test_mixed_list_xt_payload_survives_fit_for_random_effect_penalties():
    """
    Regression coverage verifying that mixed list xt payload survives fit for random
    effect penalties.
    """
    data = pd.DataFrame(
        {
            "y": np.linspace(0.1, 4.0, 40, dtype=np.float64),
            "g": np.asarray(["u", "v", "w", "q"] * 10, dtype=object),
        }
    )
    formula = (
        'y ~ s(g, bs="re", xt=list(1, S=[[[1.0, 0.0, 0.0, 0.0], '
        "[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]], "
        "rank=4))"
    )

    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
    )
    gam.fit(data=data)

    term = gam.compiled_model_.compiled_terms[0]
    term_spec = term.metadata["term_spec"]
    xt = term_spec["basis_options"]["xt"]
    assert xt[0] == 1
    assert xt["rank"] == 4
    assert len(xt["S"]) == 1


def test_list_kwargs_expansion_supports_non_identifier_xt_keys_for_tp():
    """
    Regression coverage verifying that list kwargs expansion supports non identifier xt
    keys for tp.
    """
    data = pd.DataFrame(
        {
            "y": np.linspace(0.1, 4.0, 40, dtype=np.float64),
            "x": np.linspace(0.0, 1.0, 40, dtype=np.float64),
        }
    )
    gam = GAM(
        family="gaussian",
        formula='y ~ s(x, bs="tp", k=5, xt=list(**{"max.knots": 10}, seed=2))',
        optimize_smoothing=False,
        smoothing_method="fixed",
    )
    gam.fit(data=data)

    term = gam.compiled_model_.compiled_terms[0]
    term_spec = term.metadata["term_spec"]
    assert term_spec["basis_options"]["xt"] == {"max.knots": 10, "seed": 2}
def test_tensor_xt_rejects_ambiguous_named_dict_surface():
    """
    Regression coverage verifying that tensor xt rejects ambiguous named dict surface.
    """
    data = _make_gaussian_data(seed=902, n=90)

    with pytest.raises(ValueError, match="Tensor xt dict form is only supported"):
        GAM(
            family="gaussian",
            formula='y ~ te(x0, x1, bs=["tp", "tp"], k=[8, 8], xt={"max.knots": 18, "seed": 2})',
            optimize_smoothing=False,
            smoothing_method="fixed",
        ).fit(data=data)


def test_tensor_xt_length_mismatch_raises_explicit_error():
    """
    Regression coverage verifying that tensor xt length mismatch raises explicit error.
    """
    data = _make_gaussian_data(seed=903, n=90)

    with pytest.raises(ValueError, match=r"xt must have length 1 or 2, got 3\."):
        GAM(
            family="gaussian",
            formula='y ~ ti(x0, x1, bs=["tp", "ts"], k=[8, 8], xt=[{"seed": 2}, None, None])',
            optimize_smoothing=False,
            smoothing_method="fixed",
        ).fit(data=data)


def test_pirls_laplace_reml_derivatives_dispatch_to_exact_backend(monkeypatch):
    """
    Regression coverage verifying that PIRLS laplace REML derivatives dispatch to exact
    backend.
    """
    x = np.linspace(-2.0, 2.0, 40, dtype=np.float64)
    data = pd.DataFrame({"x": x, "y": (x > 0.0).astype(np.float64)})

    gam = GAM(
        family="binomial",
        formula='y ~ s(x, bs="cr", k=8)',
        optimize_smoothing=False,
        smoothing_method="fixed",
    )
    gam.fit(data=data)
    y = gam.family.validate_y(gam.y_)

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


def test_gcv_ubre_aic_derivatives_dispatch_to_exact_backend_not_finite_difference(
    monkeypatch,
):
    """
    Regression coverage verifying that mgcv gam.fit3 GCV/UBRE/AIC derivative ports
    do not route through finite-difference fallbacks.
    """
    x = np.linspace(-2.0, 2.0, 40, dtype=np.float64)
    data = pd.DataFrame({"x": x, "y": (x > 0.0).astype(np.float64)})

    gam = GAM(
        family="binomial",
        formula='y ~ s(x, bs="cr", k=8)',
        optimize_smoothing=False,
        smoothing_method="fixed",
    )
    gam.fit(data=data)
    y = gam.family.validate_y(gam.y_)

    seen = {"fd_grad": 0, "fd_hess": 0}

    def _fd_grad(*args, **kwargs):
        del args, kwargs
        seen["fd_grad"] += 1
        return np.array([-1.0], dtype=np.float64)

    def _fd_hess(*args, **kwargs):
        del args, kwargs
        seen["fd_hess"] += 1
        return np.array([[-1.0]], dtype=np.float64)

    monkeypatch.setattr(criteria_dispatch, "criterion_gradient_numerical", _fd_grad)
    monkeypatch.setattr(criteria_dispatch, "criterion_hessian_numerical", _fd_hess)

    grad = criteria_dispatch.criterion_gradient(
        gam, y, np.log(gam.smoothing_params), method="ubre"
    )
    hess = criteria_dispatch.criterion_hessian(
        gam, y, np.log(gam.smoothing_params), method="aic"
    )

    assert seen == {"fd_grad": 0, "fd_hess": 0}
    assert np.asarray(grad, dtype=np.float64).shape == (1,)
    assert np.asarray(hess, dtype=np.float64).shape == (1, 1)
    assert np.all(np.isfinite(grad))
    assert np.all(np.isfinite(hess))


def test_direct_exact_pirls_derivative_entrypoint_runs_on_canonical_reparam_state():
    """
    Regression coverage verifying that direct exact PIRLS derivative entrypoint runs on
    canonical reparam state.
    """
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
    """
    Regression coverage verifying that tensor id metadata maps one smoothing id to
    multiple sp indices.
    """
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
    """
    Regression coverage verifying that tensor id smoothing param mapping accepts id
    keyed values.
    """
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


def test_negbin_estimated_theta_joint_path_accepts_arrays_offset_and_weights():
    """
    Regression coverage: the negbin estimated-theta joint (log theta, log sp)
    path used stale guards from a removed Rscript shim that rejected offsets,
    prior weights, and non-formula construction. Upstream gam.fit4 threads
    weights and offset through the extended-family PIRLS with no special-casing
    (mgcv/R/gam.fit4.r:240-244), and theta initialization is formula-free
    (mgcv/R/efam.r:183-193).
    """
    rng = np.random.default_rng(2024)
    n = 240
    x0 = rng.normal(size=n)
    mu = np.exp(0.2 + 0.55 * np.sin(x0))
    theta_true = 1.0
    y = rng.negative_binomial(theta_true, theta_true / (theta_true + mu), size=n)
    offset = rng.uniform(-0.1, 0.1, size=n)
    weights = rng.uniform(0.5, 1.5, size=n)

    gam = GAM(
        family={"name": "negbin", "theta": 1.8, "estimate_theta": True},
        basis="cr",
        k=8,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    )
    gam.fit(X=x0.reshape(-1, 1), y=y, offset=offset, sample_weight=weights)

    assert gam.fit_result() is not None
    theta_hat = float(np.asarray(gam.family.theta, dtype=np.float64))
    assert np.isfinite(theta_hat) and theta_hat > 0.0
    assert theta_hat != 1.8


def test_rank_deficient_gaussian_fit_matches_mgcv_drop_gauge():
    """
    Exactly aliased parametric columns must reproduce mgcv's rank-deficiency
    gauge: dropped canonical coordinates give an exactly-zero coefficient AND
    an exactly-zero Vp row/column at the same position
    (mgcv/src/gdi.c:2253-2292 zero-fill + rV scatter). Side conditions are
    disabled so the alias reaches the solver drop path as it does upstream.
    """
    from tests.mgcv_parity_utils import _run_mgcv_snapshot

    rng = np.random.default_rng(42)
    n = 90
    x0 = rng.uniform(size=n)
    x1 = rng.uniform(size=n)
    data = pd.DataFrame({"x0": x0, "x1": x1, "z": x1})
    data["y"] = (
        np.sin(2.0 * np.pi * x0) + 0.5 * x1 + 0.1 * rng.standard_normal(n)
    )
    formula = 'y ~ x1 + z + s(x0, bs="cr", k=8)'

    expected = _run_mgcv_snapshot(
        data=data, formula=formula, family="gaussian", method="REML"
    )
    exp_coef = np.asarray(expected["fit"]["coef_full"], dtype=np.float64)
    exp_vp = np.asarray(expected["fit"]["cov_bayes"], dtype=np.float64)

    gam = GAM(
        formula=formula,
        family="gaussian",
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
        apply_side_conditions=False,
    )
    gam.fit(data=data)
    fr = gam.fit_result()
    coef = np.asarray(fr.coef_full, dtype=np.float64)
    vp = np.asarray(fr.cov_bayes, dtype=np.float64)

    assert coef.shape == exp_coef.shape
    dropped = np.flatnonzero(np.diag(exp_vp) == 0.0)
    assert dropped.size == 1

    np.testing.assert_allclose(coef, exp_coef, rtol=0.0, atol=1e-8)
    assert coef[dropped[0]] == 0.0
    # The dropped coordinate must be zeroed in Vp at the SAME position as
    # mgcv (previously the covariance came from a second natural-design QR
    # whose drop landed on a different column).
    assert np.all(vp[dropped[0], :] == 0.0)
    assert np.all(vp[:, dropped[0]] == 0.0)
    np.testing.assert_allclose(vp, exp_vp, rtol=1e-6, atol=1e-9)

    np.testing.assert_allclose(
        np.asarray(gam.predict(data), dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        rtol=0.0,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    ("family_name", "seed"),
    [("poisson", 43), ("binomial", 44), ("gamma", 45)],
)
def test_rank_deficient_pirls_fit_matches_mgcv_drop_gauge(family_name, seed):
    """PIRLS must use gdi1's dropped-coordinate gauge for coef and Vp."""
    from tests.mgcv_parity_utils import _run_mgcv_snapshot

    rng = np.random.default_rng(seed)
    n = 110
    x0 = rng.uniform(size=n)
    x1 = rng.uniform(size=n)
    data = pd.DataFrame({"x0": x0, "x1": x1, "z": x1})
    eta = 0.15 + 0.55 * np.sin(2.0 * np.pi * x0) + 0.4 * x1
    mu = np.exp(eta)
    if family_name == "poisson":
        data["y"] = rng.poisson(mu)
    elif family_name == "binomial":
        data["y"] = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
    else:
        data["y"] = rng.gamma(shape=4.0, scale=mu / 4.0)
    formula = 'y ~ x1 + z + s(x0, bs="cr", k=8)'

    expected = _run_mgcv_snapshot(
        data=data, formula=formula, family=family_name, method="REML"
    )
    exp_coef = np.asarray(expected["fit"]["coef_full"], dtype=np.float64)
    exp_vp = np.asarray(expected["fit"]["cov_bayes"], dtype=np.float64)

    gam = GAM(
        formula=formula,
        family=family_name,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
        apply_side_conditions=False,
    )
    gam.fit(data=data)
    fr = gam.fit_result()
    coef = np.asarray(fr.coef_full, dtype=np.float64)
    vp = np.asarray(fr.cov_bayes, dtype=np.float64)

    dropped = np.flatnonzero(np.diag(exp_vp) == 0.0)
    assert dropped.size == 1
    np.testing.assert_array_equal(np.flatnonzero(np.diag(vp) == 0.0), dropped)
    np.testing.assert_allclose(coef, exp_coef, rtol=0.0, atol=1e-8)
    assert coef[dropped[0]] == 0.0
    assert np.all(vp[dropped[0], :] == 0.0)
    assert np.all(vp[:, dropped[0]] == 0.0)
    np.testing.assert_allclose(vp, exp_vp, rtol=2e-6, atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(gam.predict(data), dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        rtol=0.0,
        atol=1e-8,
    )
