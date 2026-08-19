from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from nampy.gam import GAM
from nampy.gam.compiler.structures import (
    CompiledModel,
    CompiledPenalty,
    CompiledPredictor,
)
from nampy.gam.fit.solvers.gamlss_utils import gamlss_etamu, trind_generator
from nampy.gam.fit.solvers.general_family.fixed_smoothing import (
    build_general_family_setup_state,
    criterion_gradient_ml_reml_general_family,
    criterion_hessian_ml_reml_general_family,
    run_general_family_fixed_smoothing,
)
from nampy.gam.smoothing_selection.reparam import _stable_penalty_logdet_derivatives

# ======================================================================
# gaulss
# ======================================================================

# ---------------------------------------------------------------------------
# 1. trind_generator
# ---------------------------------------------------------------------------


def test_trind_generator_k2_symmetry():
    """Verify that trind generator k2 symmetry."""
    tri = trind_generator(2)
    i2 = tri["i2"]
    # i2 must be symmetric
    assert i2[0, 1] == i2[1, 0]
    # packed order: (0,0)=0, (0,1)=1, (1,1)=2
    assert i2[0, 0] == 0
    assert i2[0, 1] == 1
    assert i2[1, 1] == 2


def test_trind_generator_k2_reverse():
    """Verify that trind generator k2 reverse."""
    tri = trind_generator(2)
    i2r = tri["i2r"]
    K = 2
    # i2r[m] should encode (k, l) as l + k*K
    # for K=2: packed entries are (0,0),(0,1),(1,1) → indices 0,1,3
    assert i2r[0] == 0 + 0 * K  # k=0, l=0
    assert i2r[1] == 1 + 0 * K  # k=0, l=1
    assert i2r[2] == 1 + 1 * K  # k=1, l=1


def test_trind_generator_k3_counts():
    """Verify that trind generator k3 counts."""
    tri = trind_generator(3)
    i2 = tri["i2"]
    i3 = tri["i3"]
    # K=3: K*(K+1)/2 = 6 packed second-order entries → max index = 5
    assert int(i2.max()) == 5
    # K*(K+1)*(K+2)/6 = 10 packed third-order entries → max index = 9
    assert int(i3.max()) == 9


# ---------------------------------------------------------------------------
# 2. gamlss_etamu with identity links (ig1=1, g2=g3=g4=0)
# ---------------------------------------------------------------------------


def test_gamlss_etamu_identity_links():
    """With identity links ig1=1, g2=g3=g4=0, eta-derivs == mu-derivs."""
    rng = np.random.default_rng(0)
    n, K = 50, 2
    l1 = rng.standard_normal((n, K))
    l2 = rng.standard_normal((n, 3))  # K*(K+1)/2 = 3
    l3 = rng.standard_normal((n, 4))  # K*(K+1)*(K+2)/6 = 4

    ig1 = np.ones((n, K), dtype=np.float64)  # d mu / d eta = 1
    g2 = np.zeros((n, K), dtype=np.float64)
    g3 = np.zeros((n, K), dtype=np.float64)

    tri = trind_generator(K)
    i2, i3 = tri["i2"], tri["i3"]

    de = gamlss_etamu(l1, l2, l3, 0, ig1, g2, g3, 0, i2, i3, None, deriv=1)

    assert_allclose(de["l1"], l1)
    assert_allclose(de["l2"], l2)
    assert_allclose(de["l3"], l3)


# ======================================================================
# General Family API
# ======================================================================


def test_general_family_outer_derivatives_require_exact_family_support():
    """Verify that general family outer derivatives require exact family support."""

    class _Family:
        supports_analytic_outer_derivatives = False
        supports_analytic_outer_gradient = False
        supports_analytic_outer_hessian = False

    class _Model:
        family = _Family()
        prior_weights_ = np.ones(3, dtype=np.float64)
        smoothing_params = np.ones(2, dtype=np.float64)
        smoothing_fixed_mask_ = None
        min_sp_ = None
        compiled_model_ = CompiledModel(
            predictors=(),
            design_matrix=np.empty((0, 0), dtype=np.float64),
            compiled_terms=(),
            compiled_penalties=(),
            metadata={},
            n_coef=0,
            n_smoothing_params=2,
            predictor_full_slices=(),
            coef_reduced_to_full_idx=np.empty((0,), dtype=int),
        )

    model = _Model()
    y = np.ones(3, dtype=np.float64)
    log_sp = np.array([0.0, 0.5], dtype=np.float64)

    with pytest.raises(NotImplementedError, match="analytic outer gradients"):
        criterion_gradient_ml_reml_general_family(model, y, log_sp, "REML")

    with pytest.raises(NotImplementedError, match="analytic outer Hessians"):
        criterion_hessian_ml_reml_general_family(model, y, log_sp, "REML")


def test_general_fit5_run_uses_canonical_penalty_logdet_derivatives(monkeypatch):
    """Verify that general fit5 run uses canonical penalty logdet derivatives."""
    recorded = {}

    def _stub_gam_fit5(
        _X,
        _y,
        _jj,
        _lsp,
        _St,
        _S_blocks,
        *,
        ldetS,
        ldetS1,
        ldetS2,
        Sl=None,
        **_kwargs,
    ):
        recorded["X"] = np.asarray(_X, dtype=np.float64).copy()
        recorded["Sl"] = Sl
        recorded["ldetS"] = float(ldetS)
        recorded["ldetS1"] = np.asarray(ldetS1, dtype=np.float64).copy()
        recorded["ldetS2"] = np.asarray(ldetS2, dtype=np.float64).copy()
        return {"score": 0.0}

    class _Pred:
        def __init__(self):
            self.design_matrix = np.arange(12, dtype=np.float64).reshape(4, 3)
            self.has_intercept = False

    class _Penalty:
        def __init__(self):
            self.coef_slice = slice(0, 3)
            self.smoothing_index = 0
            self.matrix = np.diag([1.0, 2.0, 3.0])

    class _Model:
        n_samples_ = 4
        max_irls_iter = 2
        irls_tol = 1e-7
        hparams = {}
        prior_weights_ = np.ones(4, dtype=np.float64)
        compiled_model_ = CompiledModel(
            predictors=(
                CompiledPredictor(
                    name="eta1",
                    design_matrix=_Pred().design_matrix,
                    compiled_terms=(),
                    compiled_penalties=(),
                    smoothing_parameter_map={},
                    n_coef=3,
                    n_smoothing_params=1,
                    has_intercept=False,
                ),
            ),
            design_matrix=_Pred().design_matrix,
            compiled_terms=(),
            compiled_penalties=(
                CompiledPenalty(
                    label="s(x)",
                    coef_slice=slice(0, 3),
                    matrix=_Penalty().matrix,
                    smoothing_index=0,
                ),
            ),
            metadata={},
            n_coef=3,
            n_smoothing_params=1,
            predictor_full_slices=(slice(0, 3),),
            coef_reduced_to_full_idx=np.arange(3, dtype=int),
        )
        family = object()
        _optim_method = "REML"
        smoothing_params = np.ones(1, dtype=np.float64)
        smoothing_fixed_mask_ = None
        min_sp_ = None

    monkeypatch.setattr(
        "nampy.gam.smoothing_selection.reparam._stable_penalty_logdet_derivatives",
        lambda *_args, **_kwargs: (
            3.5,
            np.array([1.0], dtype=np.float64),
            np.array([[7.0]], dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        "nampy.gam.fit.solvers.general_family.newton.solve_general_newton_fit",
        _stub_gam_fit5,
    )

    model = _Model()
    setup = build_general_family_setup_state(model, np.array([2.0]), score_type="REML")
    assert len(setup.Sl) == 1
    block = setup.Sl[0]
    assert block.start == 1
    assert block.stop == 3
    assert block.linear is True
    assert block.repara is True
    assert block.rank == 3
    np.testing.assert_allclose(
        block.S[0],
        _Penalty().matrix,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(block.ind, np.array([True, True, True]))
    np.testing.assert_allclose(
        block.D,
        np.array([1.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(3.0)], dtype=np.float64),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        setup.Sl.E,
        np.eye(3, dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        setup.Sl.S,
        np.eye(3, dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        setup.Sl.lambda_,
        np.array([1.0], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(setup.X_full, _Pred().design_matrix, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        setup.X_initial,
        _Pred().design_matrix
        * np.array([1.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(3.0)], dtype=np.float64)[
            np.newaxis, :
        ],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        setup.St,
        2.0 * _Penalty().matrix,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        setup.S_blocks[0],
        _Penalty().matrix,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        setup.log_sp, np.log(np.array([2.0])), rtol=0.0, atol=0.0
    )
    assert setup.Mp == 0
    assert setup.score_type == "REML"
    assert len(setup.jj) == 1
    np.testing.assert_array_equal(setup.jj[0], np.array([0, 1, 2], dtype=int))

    run = run_general_family_fixed_smoothing(
        model, np.ones(4, dtype=np.float64), np.array([2.0])
    )

    assert recorded["ldetS"] == pytest.approx(3.5)
    np.testing.assert_allclose(recorded["ldetS1"], np.array([1.0], dtype=np.float64))
    np.testing.assert_allclose(recorded["ldetS2"], np.array([[7.0]], dtype=np.float64))
    np.testing.assert_allclose(
        run["setup"].X_initial, recorded["X"], rtol=0.0, atol=0.0
    )
    assert recorded["Sl"] is run["setup"].Sl
    np.testing.assert_allclose(run["setup"].St, setup.St, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(run["setup"].log_sp, setup.log_sp, rtol=0.0, atol=0.0)
    assert run["setup"].Mp == setup.Mp


def test_general_fit5_penalty_logdet_derivatives_match_finite_difference():
    """Verify that general fit5 penalty logdet derivatives match finite difference."""
    rng = np.random.default_rng(123)
    n = 80
    x = np.linspace(-1.0, 1.0, n)
    mu = 0.3 + 0.5 * x
    sigma = np.exp(-0.2 + 0.1 * x)
    y = rng.normal(mu, sigma, size=n)
    data = pd.DataFrame({"y": y, "x": x})

    gam = GAM(
        family="gaulss",
        formula=['y ~ s(x, bs="cr", k=6)', "~ 1"],
        optimize_smoothing=True,
        smoothing_method="REML",
    )
    gam.fit(data=data)

    sp = np.asarray(gam.smoothing_params, dtype=np.float64).ravel()
    log_sp = np.log(np.clip(sp, 1e-300, None))
    logdet, grad, hess = _stable_penalty_logdet_derivatives(gam, sp, order=2)
    ref_logdet, ref_grad, ref_hess = _stable_penalty_logdet_derivatives(
        gam, sp, order=2
    )

    np.testing.assert_allclose(logdet, ref_logdet, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(grad, ref_grad, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(hess, ref_hess, rtol=0.0, atol=1e-12)

    steps = np.maximum(1e-4, 1e-3 * (1.0 + np.abs(log_sp)))
    fd_grad = np.zeros_like(grad)
    fd_hess = np.zeros_like(hess)

    for j, h in enumerate(steps):
        rho_plus = log_sp.copy()
        rho_minus = log_sp.copy()
        rho_plus[j] += h
        rho_minus[j] -= h
        sp_plus = np.exp(rho_plus)
        sp_minus = np.exp(rho_minus)

        val_plus = _stable_penalty_logdet_derivatives(gam, sp_plus, order=2)[0]
        val_minus = _stable_penalty_logdet_derivatives(gam, sp_minus, order=2)[0]
        fd_grad[j] = (val_plus - val_minus) / (2.0 * h)

        grad_plus = _stable_penalty_logdet_derivatives(gam, sp_plus, order=2)[1]
        grad_minus = _stable_penalty_logdet_derivatives(gam, sp_minus, order=2)[1]
        fd_hess[:, j] = (grad_plus - grad_minus) / (2.0 * h)

    fd_hess = 0.5 * (fd_hess + fd_hess.T)

    np.testing.assert_allclose(grad, fd_grad, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(hess, fd_hess, rtol=2e-4, atol=5e-5)
