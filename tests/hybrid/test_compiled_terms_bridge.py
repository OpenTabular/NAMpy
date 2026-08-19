"""Compiled-GAM-terms Torch bridge: fidelity to the numpy compiler stage.

The bridge is NOT an mgcv fit (fixed lambda, Torch optimization) and is
never parity-compared; these tests pin that the Torch side reproduces the
compiled numpy quantities exactly and that the fixed-lambda quadratic
objective converges to its closed-form penalized least-squares solution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from nampy.gam import GAM
from nampy.hybrid import (
    CompiledGAMTerms,
    CompiledGAMTermsModule,
    GAMNetRegressor,
)
from nampy.neural.configs.linreg_config import DefaultLinRegConfig
from nampy.neural.modules.linreg import LinReg

_FIT_KWARGS = {
    "max_epochs": 80,
    "patience": 80,
    "lr": 5e-2,
    "batch_size": 64,
    "logger": False,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "num_sanity_val_steps": 0,
}


def _data(n=180, seed=0):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({"x0": rng.uniform(size=n), "x3": rng.normal(size=n)})
    data["y"] = (
        np.sin(3.0 * data["x0"])
        + 1.5 * data["x3"]
        + rng.normal(scale=0.1, size=n)
    )
    return data


def _terms(data, lam=1.0):
    return CompiledGAMTerms.from_formula("y ~ s(x0, k=6)", data, lam=[lam])


def test_penalty_matches_numpy_for_random_beta():
    terms = _terms(_data())
    module = CompiledGAMTermsModule(terms)
    torch.manual_seed(0)
    with torch.no_grad():
        module.beta.copy_(torch.randn(terms.n_coef, 1))

    beta = module.beta.detach().numpy()[:, 0]
    expected = sum(
        lam * beta[coef_slice] @ matrix @ beta[coef_slice]
        for coef_slice, matrix, lam in terms.penalty_payload()
    )
    assert float(module.penalty().detach()) == pytest.approx(
        expected, rel=1e-5
    )


def test_design_fidelity_on_training_and_new_data():
    data = _data()
    terms = _terms(data)
    module = CompiledGAMTermsModule(terms)
    torch.manual_seed(1)
    with torch.no_grad():
        module.beta.copy_(torch.randn(terms.n_coef, 1))

    design = terms.design(None)
    output = module(torch.tensor(design, dtype=torch.float32))["output"]
    beta = module.beta.detach().numpy()[:, 0]
    manual = design @ beta + float(module.intercept.detach())
    np.testing.assert_allclose(
        output.detach().numpy()[:, 0], manual, atol=1e-4
    )

    # Compiling and predicting on the same rows must agree exactly.
    np.testing.assert_allclose(terms.design(data), design, atol=1e-10)


def test_gradient_flows_to_beta():
    terms = _terms(_data())
    module = CompiledGAMTermsModule(terms)
    design = torch.tensor(terms.design(None), dtype=torch.float32)
    result = module(design)
    loss = result["output"].pow(2).mean() + result["gam_penalty"]
    # beta starts at zero, so drive it through the penalty-free data term.
    with torch.no_grad():
        module.beta.add_(0.1)
    result = module(design)
    loss = result["output"].pow(2).mean() + result["gam_penalty"]
    loss.backward()
    assert module.beta.grad is not None
    assert float(module.beta.grad.abs().sum()) > 0.0


def test_lam_length_mismatch_raises():
    data = _data()
    with pytest.raises(ValueError, match="lam must have length"):
        CompiledGAMTerms.from_formula("y ~ s(x0, k=6)", data, lam=[1.0, 2.0])
    with pytest.raises(ValueError, match="non-negative"):
        CompiledGAMTerms.from_formula("y ~ s(x0, k=6)", data, lam=[-1.0])


def test_raw_basis_guard_raises():
    data = _data(n=120)
    gam = GAM(
        formula="y ~ s(x0, k=5)",
        family="gaussian",
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=[1.0],
    )
    gam.fit(data=data)
    gam.compiled_model_.compiled_terms[0].metadata[
        "expose_raw_prediction_basis"
    ] = True
    try:
        with pytest.raises(NotImplementedError, match="raw prediction basis"):
            CompiledGAMTerms.from_fitted_gam(gam)
    finally:
        del gam.compiled_model_.compiled_terms[0].metadata[
            "expose_raw_prediction_basis"
        ]


def test_from_fitted_gam_lifts_without_modifying():
    data = _data(n=150)
    gam = GAM(
        formula="y ~ s(x0, k=6)",
        family="gaussian",
        optimize_smoothing=True,
        smoothing_method="reml",
    )
    gam.fit(data=data)
    coef_before = gam.fit_result().coef_full.copy()

    terms = CompiledGAMTerms.from_fitted_gam(gam)
    np.testing.assert_array_equal(
        terms.lam, gam.fit_result().smoothing_params
    )
    assert terms.compiled_model is gam.compiled_model_
    np.testing.assert_array_equal(gam.fit_result().coef_full, coef_before)

    # Design for the training frame matches the compiled training design.
    np.testing.assert_allclose(
        terms.design(data), terms.design(None), atol=1e-10
    )


def test_adam_converges_to_penalized_least_squares():
    data = _data(n=150, seed=2)
    terms = _terms(data, lam=0.5)
    design = terms.design(None)
    y = terms.response.astype(np.float64)
    n, p = design.shape

    # Closed-form minimizer of mean((B_full b - y)^2) + b' P_full b with the
    # intercept column prepended and penalties placed at +1 offset.
    B = np.column_stack([np.ones(n), design])
    P = np.zeros((p + 1, p + 1))
    for coef_slice, matrix, lam in terms.penalty_payload():
        start = coef_slice.start + 1
        stop = coef_slice.stop + 1
        P[start:stop, start:stop] += lam * matrix
    closed_form = np.linalg.solve(B.T @ B + n * P, B.T @ y)

    module = CompiledGAMTermsModule(terms)
    design_t = torch.tensor(design, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    optimizer = torch.optim.Adam(module.parameters(), lr=5e-2)
    for _ in range(800):
        optimizer.zero_grad()
        result = module(design_t)
        loss = (result["output"] - y_t).pow(2).mean() + result["gam_penalty"]
        loss.backward()
        optimizer.step()

    fitted = np.concatenate(
        [
            module.intercept.detach().numpy(),
            module.beta.detach().numpy()[:, 0],
        ]
    )
    np.testing.assert_allclose(B @ fitted, B @ closed_form, atol=0.05)


def test_hybrid_joint_regressor_end_to_end(tmp_path):
    data = _data()
    estimator = GAMNetRegressor(
        "y ~ s(x0, k=6)",
        LinReg,
        DefaultLinRegConfig,
        lam=[0.5],
        numerical_preprocessing="standardization",
    )
    kwargs = dict(_FIT_KWARGS, checkpoint_path=str(tmp_path))
    estimator.fit(data, neural_features=["x3"], **kwargs)

    assert estimator.score(data, data["y"]) > 0.85

    result = estimator._predict(data)
    assert "gam_baseline" in result
    assert "gam_penalty" in result

    path = estimator.save_model(tmp_path / "joint.nampy")
    assert path.stat().st_size < 5_000_000  # no design/checkpoint bloat
    restored = GAMNetRegressor.load_model(path)
    np.testing.assert_allclose(
        restored.predict(data), estimator.predict(data), atol=1e-6
    )


@pytest.mark.parametrize("lam", [1e-4, 1.0, 1e4])
def test_penalty_exact_across_lambda_scales(lam):
    terms = _terms(_data(), lam=lam)
    module = CompiledGAMTermsModule(terms)
    torch.manual_seed(2)
    with torch.no_grad():
        module.beta.copy_(torch.randn(terms.n_coef, 1))

    beta = module.beta.detach().numpy()[:, 0]
    expected = sum(
        lam_value * beta[coef_slice] @ matrix @ beta[coef_slice]
        for coef_slice, matrix, lam_value in terms.penalty_payload()
    )
    assert float(module.penalty().detach()) == pytest.approx(expected, rel=1e-4)
