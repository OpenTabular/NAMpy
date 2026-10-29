"""Exact mixed-response joint likelihood and public zero-adjusted Gamma routes."""

from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit
from sklearn.base import clone

from nampy import GAMLSS
from nampy.diagnostics import quantile_residuals
from nampy.gam.families.gamlss._taylor import Taylor
from nampy.gam.families.gamlss.semicontinuous import ZeroAdjustedGamma
from nampy.gam.model_state import _fit_result


def _data():
    rng = np.random.default_rng(222)
    x = np.linspace(-1, 1, 180)
    mu, sigma, pi0 = np.exp(0.4 + 0.3 * x), 0.55, expit(-0.8 + 0.4 * x)
    y = rng.gamma(1 / sigma**2, mu * sigma**2)
    y[rng.uniform(size=len(x)) < pi0] = 0
    return pd.DataFrame({"x": x, "y": y})


def _model(*, smooth=False, **kwargs):
    formula = {
        "mu": 'y ~ s(x,bs="cr",k=5)' if smooth else "y ~ x",
        "sigma": "~ 1",
        "pi0": '~ s(x,bs="cr",k=4)' if smooth else "~ x",
    }
    return GAMLSS(family="zaga_lss", formula=formula, **kwargs)


def test_unpenalized_fit_matches_independent_normalized_mle():
    frame = _data()
    weights = np.tile([0.0, 1.0, 2.0], 60)
    offsets = [
        0.1 * np.sin(frame.x.to_numpy()),
        np.full(len(frame), 0.05),
        np.linspace(-0.1, 0.1, len(frame)),
    ]
    model = _model(optimize_smoothing=False).fit(frame, sample_weight=weights, offset=np.column_stack(offsets))
    assert _fit_result(model.gam_).converged
    design = np.column_stack([np.ones(len(frame)), frame.x])
    positive = frame.y.to_numpy() > 0

    def objective(beta):
        mu, sigma = np.exp(design @ beta[:2] + offsets[0]), np.exp(beta[2] + offsets[1])
        logit = design @ beta[3:] + offsets[2]
        likelihood = -np.logaddexp(0, -logit)
        likelihood[positive] = -np.logaddexp(0, logit[positive]) + stats.gamma.logpdf(
            frame.y[positive],
            1 / sigma[positive] ** 2,
            scale=mu[positive] * sigma[positive] ** 2,
        )
        return -np.sum(weights * likelihood)

    oracle = minimize(
        objective,
        [0.4, 0.3, np.log(0.55), -0.8, 0.4],
        method="BFGS",
        options={"gtol": 1e-6},
    )
    parameters = model.predict_parameters(frame, offset=np.column_stack(offsets))
    assert_allclose(
        parameters[:, 0], np.exp(design @ oracle.x[:2] + offsets[0]), atol=3e-6
    )
    assert_allclose(parameters[:, 1], np.exp(oracle.x[2] + offsets[1]), atol=3e-6)
    assert_allclose(
        parameters[:, 2], expit(design @ oracle.x[3:] + offsets[2]), atol=3e-6
    )
    distribution = model.predict_distribution(frame, offset=np.column_stack(offsets))
    assert_allclose(
        model.predict_point(frame, offset=np.column_stack(offsets)), distribution.mean(), atol=1e-12
    )
    assert_allclose(distribution.mean(), (1 - parameters[:, 2]) * parameters[:, 0])
    assert_allclose(
        distribution.variance(),
        (1 - parameters[:, 2])
        * parameters[:, 0] ** 2
        * (parameters[:, 1] ** 2 + parameters[:, 2]),
    )
    assert np.isfinite(
        quantile_residuals(distribution, frame.y, random_state=1).residuals
    ).all()


def test_fixed_smoothing_weight_duplication_uncertainty_and_persistence(tmp_path):
    frame = _data()
    weights = np.tile([1, 2, 3], 60)
    model = _model(
        smooth=True,
        optimize_smoothing=False,
        smoothing_params=[0.3, 0.7],
        irls_tol=1e-11,
    ).fit(frame, sample_weight=weights)
    index = np.repeat(np.arange(len(frame)), weights)
    repeated = _model(
        smooth=True,
        optimize_smoothing=False,
        smoothing_params=[0.3, 0.7],
        irls_tol=1e-11,
    ).fit(frame.iloc[index].reset_index(drop=True))
    assert_allclose(model.predict_parameters(frame), repeated.predict_parameters(frame), atol=3e-6)
    components = model.predict_components(frame)
    components.validate_additive_reconstruction()
    assert components.link.shape == (len(frame), 3)
    assert model.parameter_names_ == ("mu", "sigma", "pi0")
    assert np.isfinite(model.standard_errors(frame)).all()
    assert clone(model).family == "zaga_lss"
    path = tmp_path / "zaga.pkl"
    model.save_model(path)
    restored = GAMLSS.load_model(path)
    assert_allclose(
        restored.predict_distribution(frame).quantile([0.1, 0.5, 0.9]),
        model.predict_distribution(frame).quantile([0.1, 0.5, 0.9]),
    )
    missing = frame.copy()
    missing.loc[3, "x"] = np.nan
    assert np.isnan(model.predict_distribution(missing).mean()[3])


@pytest.mark.parametrize("eta", [[0.3, -0.4, -0.7], [-0.2, 0.3, 0.9]])
def test_exact_three_predictor_tensors_including_atom_rows_against_mpmath(eta):
    mp = pytest.importorskip("mpmath")
    y = np.array([0.0, 0.8, 2.0])
    family = ZeroAdjustedGamma()
    variables = [Taylor(np.full(len(y), eta[k]), 3, 4, k) for k in range(3)]
    value = family._log_density(y, *variables)
    with mp.workdps(60):
        point = tuple(mp.mpf(str(x)) for x in eta)
        for row, response in enumerate(y):

            def likelihood(lm, ls, lz, response=response):
                if response == 0:
                    return -mp.log1p(mp.exp(-lz))
                shape = mp.exp(-2 * ls)
                scale = mp.exp(lm + 2 * ls)
                yy = mp.mpf(str(response))
                return (
                    (shape - 1) * mp.log(yy)
                    - yy / scale
                    - shape * mp.log(scale)
                    - mp.loggamma(shape)
                    - mp.log1p(mp.exp(lz))
                )

            assert_allclose(value.value[row], float(likelihood(*point)), atol=2e-12)
            for degree in range(1, 5):
                indices = list(combinations_with_replacement(range(3), degree))
                reference = [
                    float(
                        mp.diff(
                            likelihood, point, tuple(idx.count(k) for k in range(3))
                        )
                    )
                    for idx in indices
                ]
                assert_allclose(
                    value.packed_derivatives(degree)[row],
                    reference,
                    rtol=2e-11,
                    atol=2e-11,
                )
                if row == 0:
                    for j, idx in enumerate(indices):
                        if 0 in idx or 1 in idx:
                            assert value.packed_derivatives(degree)[0, j] == 0


@pytest.mark.parametrize("value", [-1000.0, -20.0, 0.0, 20.0, 1000.0])
def test_softplus_taylor_is_stable_and_matches_torch_through_fourth_order(value):
    torch = pytest.importorskip("torch")
    jet = Taylor(np.array([value]), 1, 4, 0).softplus()
    # Branch before differentiating: avoid overflowing unused exponentials and
    # Torch softplus's linearization threshold, including its boundary at 20.
    function = (
        (lambda z: torch.log1p(torch.exp(z)))
        if value <= 0
        else (lambda z: z + torch.log1p(torch.exp(-z)))
    )
    z = torch.tensor(value, dtype=torch.float64)
    assert_allclose(jet.value, function(z).numpy(), atol=1e-14)
    for degree in range(1, 5):
        function = torch.func.jacfwd(function)
        assert_allclose(
            jet.packed_derivatives(degree)[0, 0], function(z).numpy(), atol=3e-15
        )


@pytest.mark.parametrize(
    "kind",
    ["negative", "all_zero", "all_positive", "constant_positive", "zero_weight_zeros"],
)
def test_absent_or_degenerate_components_are_rejected(kind):
    frame = _data()
    weights = np.ones(len(frame))
    if kind == "negative":
        frame.loc[0, "y"] = -1
    elif kind == "all_zero":
        frame.y = 0.0
    elif kind == "all_positive":
        frame.y = np.maximum(frame.y, 0.1)
    elif kind == "constant_positive":
        frame.loc[frame.y > 0, "y"] = 1.0
    else:
        weights[frame.y == 0] = 0
    with pytest.raises(ValueError, match="nonnegative|components"):
        _model(optimize_smoothing=False).fit(frame, sample_weight=weights)


def test_saved_r_zaga_semantics_and_unpenalized_fit():
    root = Path(__file__).parents[1] / "reference_fixtures" / "zero_adjusted_gamma"
    ref = pd.read_csv(root / "probability.csv")
    from nampy.gam.predict.distribution_adapter import distribution_prediction

    distribution = distribution_prediction(
        ZeroAdjustedGamma(), ref[["mu", "sigma", "pi0"]].to_numpy()
    )
    assert_allclose(distribution.log_prob(ref.y), ref.log_prob, atol=2e-13)
    assert_allclose(distribution.cdf(ref.y), ref.cdf, atol=2e-13)
    for i, p in enumerate(ref.probability):
        assert_allclose(
            distribution.quantile(p)[i, 0], ref["quantile"][i], rtol=3e-12, atol=1e-14
        )
    frame = pd.read_csv(root / "fit.csv")
    model = _model(optimize_smoothing=False).fit(frame)
    assert _fit_result(model.gam_).converged
    assert_allclose(
        model.predict_parameters(frame), frame[["mu", "sigma", "pi0"]], atol=3e-5, rtol=0
    )


@pytest.mark.parametrize("weighted", [False, True])
def test_outer_reml_gradient_hessian_with_offsets(weighted):
    from nampy.gam.fit.selection.criteria import (
        criterion_gradient,
        criterion_hessian,
        criterion_value,
    )

    frame = _data()
    weights = np.tile([0.0, 1.0, 2.0], 60) if weighted else None
    offsets = [
        0.1 * np.sin(frame.x.to_numpy()),
        np.full(len(frame), 0.05),
        np.linspace(-0.1, 0.1, len(frame)),
    ]
    model = _model(smooth=True, optimize_smoothing=False, smoothing_params=[0.3, 0.7], irls_tol=1e-09).fit(frame, sample_weight=weights, offset=np.column_stack(offsets))
    gam = model.gam_
    rho = np.log(gam.smoothing_params)
    gradient = criterion_gradient(gam, gam.y_, rho, method="reml")
    hessian = criterion_hessian(gam, gam.y_, rho, method="reml")
    assert_allclose(
        gradient,
        approx_derivative(
            lambda z: np.array([criterion_value(gam, gam.y_, z, method="reml")]), rho
        ).ravel(),
        atol=2e-4,
        rtol=2e-4,
    )
    assert_allclose(
        hessian,
        approx_derivative(
            lambda z: criterion_gradient(gam, gam.y_, z, method="reml"), rho
        ),
        atol=2e-3,
        rtol=2e-3,
    )


@pytest.mark.parametrize("optimizer", ["outer_newton", "bfgs", "efs"])
def test_validated_automatic_general_reml_routes(optimizer):
    frame = _data()
    model = _model(smooth=True, smoothing_optimizer=optimizer).fit(frame)
    assert _fit_result(model.gam_).converged
    assert model.gam_._optim_method == "reml"
    assert np.all(np.isfinite(model.gam_.smoothing_params))
    assert np.all(np.asarray(model.gam_.smoothing_params) > 0)
    assert np.isfinite(model.score(frame, frame.y))


def test_unavailable_optimizers_and_links_fail_explicitly():
    from nampy.gam.families.registry import make_gam_family

    for optimizer in ["magic", "nlm.fd", "optim"]:
        with pytest.raises(NotImplementedError):
            _model(smooth=True, smoothing_optimizer=optimizer).fit(_data())
    with pytest.raises(ValueError, match="Unsupported"):
        make_gam_family({"name": "zaga_lss", "link": "identity"})
