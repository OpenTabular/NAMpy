"""Joint likelihood derivatives and R gamlss.dist parameterization parity."""

import json
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from nampy import GAMLSS
from nampy.diagnostics import quantile_residuals
from nampy.gam.families.gamlss._taylor import Taylor
from nampy.gam.families.registry import make_gam_family
from nampy.gam.inference.functionals import joint_edf
from nampy.gam.model_state import _coef_full, _compiled_model, _fit_result


@pytest.fixture(scope="module")
def reference():
    return json.loads(
        (
            Path(__file__).parents[1] / "reference_fixtures/extensions/neighbors.json"
        ).read_text()
    )


@pytest.mark.parametrize("name", ["student_lss", "beta_lss", "nb_lss"])
def test_natural_parameters_density_and_mle_match_r_gamlss_dist(reference, name):
    expected = next(
        f for f in reference["upstream"]["family_fits"] if f["name"] == name
    )
    x = np.asarray(reference["inputs"]["families"]["x"])
    y = np.asarray(reference["inputs"]["families"][name])
    frame = pd.DataFrame({"x": x, "y": y})
    family = make_gam_family(name)
    second = family.parameter_names[1]
    model = GAMLSS(
        family=name, formula={"mu": "y~x", second: "~x"}, optimize_smoothing=False
    ).fit(frame)
    assert_allclose(_coef_full(model.gam_), expected["coefficients"], atol=2e-5, rtol=0)
    params = model.predict_parameters(frame)
    assert params.shape == (len(y), 2)
    assert_allclose(
        family.logpdf_from_parameters(y, params),
        expected["log_density"],
        atol=1e-5,
        rtol=0,
    )
    distribution = model.predict_distribution(frame)
    assert_allclose(
        distribution.log_prob(y), expected["log_density"], atol=1e-5, rtol=0
    )
    assert np.isfinite(
        quantile_residuals(distribution, y, random_state=7).residuals
    ).all()
    components = model.predict_components(frame)
    assert components.link.shape == (len(y), 2)
    assert np.isfinite(model.score(frame, y))
    fit = _fit_result(model.gam_)
    assert fit.converged
    assert fit.edf == pytest.approx(4, abs=1e-8)
    assert_allclose(
        family.distribution_parameters_from_eta(model.predict_link(frame)), params
    )


@pytest.mark.parametrize("name", ["student_lss", "beta_lss", "nb_lss"])
@pytest.mark.parametrize("eta", [(-3.0, -2.0), (0.2, 0.4), (3.0, 2.0)])
def test_fourth_order_joint_derivatives_against_independent_torch_autodiff(name, eta):
    torch = pytest.importorskip("torch")
    family = make_gam_family(name)
    response = {"student_lss": 0.4, "beta_lss": 0.3, "nb_lss": 3.0}[name]
    eta = np.asarray(eta)
    variables = [Taylor(eta[k : k + 1], 2, 4, k) for k in range(2)]
    jet = family._log_density(np.array([response]), *variables)

    def log_density(z):
        mu = (
            z[0]
            if name == "student_lss"
            else z[0].sigmoid()
            if name == "beta_lss"
            else z[0].exp()
        )
        scale = z[1].exp()
        if name == "student_lss":
            return torch.distributions.StudentT(5.0, mu, scale).log_prob(
                torch.tensor(response, dtype=torch.float64)
            )
        if name == "beta_lss":
            return torch.distributions.Beta(mu * scale, (1 - mu) * scale).log_prob(
                torch.tensor(response, dtype=torch.float64)
            )
        size = 1 / scale
        y = torch.tensor(response, dtype=torch.float64)
        return (
            torch.lgamma(y + size)
            - torch.lgamma(size)
            - torch.lgamma(y + 1)
            + size * (size.log() - (size + mu).log())
            + y * (mu.log() - (size + mu).log())
        )

    function = log_density
    tensor = torch.tensor(eta, dtype=torch.float64)
    assert_allclose(jet.value, function(tensor).detach().numpy(), atol=2e-8)
    for degree in range(1, 5):
        function = torch.func.jacfwd(function)
        actual = function(tensor).detach().numpy()
        packed = [
            actual[idx] for idx in combinations_with_replacement(range(2), degree)
        ]
        assert_allclose(jet.packed_derivatives(degree)[0], packed, atol=2e-8, rtol=1e-8)


@pytest.mark.parametrize("name", ["student_lss", "beta_lss", "nb_lss"])
def test_penalized_joint_edf_covariance_reml_and_serialization(
    reference, name, tmp_path
):
    from scipy.optimize._numdiff import approx_derivative

    frame = pd.DataFrame(
        {
            "x": reference["inputs"]["families"]["x"],
            "y": reference["inputs"]["families"][name],
        }
    )
    family = make_gam_family(name)
    second = family.parameter_names[1]
    model = GAMLSS(
        family=name,
        formula={"mu": "y~s(x,k=5)", second: "~s(x,k=4)"},
        optimize_smoothing=False,
        smoothing_params=[0.2, 0.4],
    ).fit(frame)
    gam = model.gam_
    compiled = _compiled_model(gam)
    X = gam.lpmatrix(frame)
    beta = _coef_full(gam)
    from nampy.gam.model_state import _predictor_full_indices

    jj = _predictor_full_indices(gam)
    weights = np.ones(len(frame))

    def score(b):
        return family.ll(frame.y.to_numpy(), X, jj, b, weights, deriv=1)["lb"]

    information = -approx_derivative(score, beta, method="3-point")
    information = (information + information.T) / 2
    S = np.zeros((len(beta), len(beta)))
    mapping = compiled.coef_reduced_to_full_idx
    for block in compiled.compiled_penalties:
        idx = mapping[block.coef_slice]
        S[np.ix_(idx, idx)] += (
            gam.smoothing_params[block.smoothing_index] * block.matrix
        )
    fit = _fit_result(gam)
    assert_allclose(joint_edf(information, S)[1], fit.edf, atol=1e-6)
    assert_allclose(np.linalg.inv(information + S), fit.cov_bayes, atol=2e-6)
    assert_allclose(np.sum(fit.edf_by_term) + 2, fit.edf, atol=1e-8)
    path = model.save_model(tmp_path / (name + ".pkl"))
    restored = GAMLSS.load_model(path)
    assert_allclose(restored.predict_parameters(frame), model.predict_parameters(frame))
    # Exercise derivatives through the outer REML criterion, not just fitting.
    from nampy.gam.fit.selection.criteria import (
        criterion_gradient,
        criterion_hessian,
        criterion_value,
    )

    log_sp = np.log(gam.smoothing_params)

    def value(rho):
        return criterion_value(gam, gam.y_, rho, method="reml")

    grad = criterion_gradient(gam, gam.y_, log_sp, method="reml")
    numeric_grad = approx_derivative(
        lambda z: np.array([value(z)]), log_sp, method="3-point"
    ).ravel()
    assert_allclose(grad, numeric_grad, atol=2e-4, rtol=2e-4)
    hessian = criterion_hessian(gam, gam.y_, log_sp, method="reml")
    numeric_hess = approx_derivative(
        lambda z: criterion_gradient(gam, gam.y_, z, method="reml"),
        log_sp,
        method="3-point",
    )
    assert_allclose(hessian, numeric_hess, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize(
    "family,response",
    [
        ("beta_lss", [0, 0.5]),
        ("beta_lss", [0.2, 1]),
        ("nb_lss", [-1, 2]),
        ("nb_lss", [0.5, 2]),
        ("student_lss", [1, np.inf]),
    ],
)
def test_support_is_validated(family, response):
    with pytest.raises(ValueError):
        make_gam_family(family).validate_y(response)


@pytest.mark.parametrize("name", ["student_lss", "beta_lss", "nb_lss"])
def test_frequency_weights_equal_duplicating_rows(reference, name):
    source = reference["inputs"]["families"]
    frame = pd.DataFrame({"x": source["x"], "y": source[name]})
    family = make_gam_family(name)
    second = family.parameter_names[1]
    weights = np.tile([1, 2, 3], len(frame) // 3)
    options = {
        "family": name,
        "formula": {"mu": "y~x", second: "~x"},
        "optimize_smoothing": False,
    }
    weighted = GAMLSS(**options).fit(frame, sample_weight=weights)
    repeated = GAMLSS(**options).fit(
        frame.loc[frame.index.repeat(weights)].reset_index(drop=True)
    )
    assert_allclose(
        weighted.predict_parameters(frame), repeated.predict_parameters(frame), atol=2e-6, rtol=1e-6
    )


@pytest.mark.parametrize("name", ["student_lss", "beta_lss", "nb_lss"])
def test_automatic_reml_fitting_route(reference, name):
    source = reference["inputs"]["families"]
    frame = pd.DataFrame({"x": source["x"], "y": source[name]})
    second = make_gam_family(name).parameter_names[1]
    model = GAMLSS(family=name, formula={"mu": "y~s(x,k=5)", second: "~s(x,k=4)"}).fit(
        frame
    )
    assert _fit_result(model.gam_).converged
    assert np.all(np.isfinite(model.predict_parameters(frame)))
    assert np.all(np.asarray(model.gam_.smoothing_params) > 0)
