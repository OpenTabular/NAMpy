"""Regression for gam.fit4 initialization through the shared weighted call."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from nampy.gam import GAM
from nampy.gam.families.betar import BetaRegressionFamily
from nampy.gam.families.negbin import NegativeBinomialLogFamily
from nampy.gam.families.ocat import OrderedCategoricalFamily
from nampy.gam.families.scat import ScaledTFamily
from nampy.gam.fit.solvers.irls_core import _use_exact_extended_family_terms, irls_core

INITIALIZERS = [
    (NegativeBinomialLogFamily(theta=2), [0, 1, 3, 5]),
    (BetaRegressionFamily(theta=5), [0.1, 0.3, 0.6, 0.9]),
    (OrderedCategoricalFamily(R=3), [1, 2, 3, 1]),
    (ScaledTFamily(theta=[5, 1]), [0, 1, 3, 5]),
]


@pytest.mark.parametrize(
    "family,y", INITIALIZERS, ids=lambda value: getattr(value, "name", None)
)
def test_exact_extended_initializers_accept_shared_weights(family, y):
    assert _use_exact_extended_family_terms(family)
    # Source initialize expressions in these families do not use weights.
    expected = family.initialize_mu(np.asarray(y, dtype=float))
    actual = family.initialize_mu(
        np.asarray(y, dtype=float), weights=np.array([0.0, 0.5, 1.0, 2.0])
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("estimate_theta", [False, True])
def test_negative_binomial_shared_cold_initializer(estimate_theta):
    x = np.linspace(-1, 1, 8)
    y = np.array([0.0, 1.0, 2.0, 0.0, 3.0, 1.0, 4.0, 2.0])
    family = NegativeBinomialLogFamily(theta=2, estimate_theta=estimate_theta)
    result = irls_core(
        X=np.column_stack([np.ones(8), x]),
        y=y,
        family=family,
        S=np.diag([0.0, 0.1]),
        weights=np.array([0.0, 0.5, 1.0, 2.0, 1.0, 0.5, 1.0, 2.0]),
        max_iter=1,
    )
    assert np.isfinite(result["coef_full"]).all()
    assert np.isfinite(result["mu"]).all()


def test_negative_binomial_count_notebook_offset_fit():
    # Exact model/data cell from docs CI40270261759, formerly failing at
    # irls_core's unconditional family.initialize_mu(y, weights=weights).
    rng = np.random.default_rng(106)
    data = pd.DataFrame(
        {"x": rng.uniform(0, 1, 900), "exposure": rng.uniform(0.5, 3, 900)}
    )
    data["log_exposure"] = np.log(data.exposure)
    mu = data.exposure * np.exp(0.7 + np.sin(2 * np.pi * data.x))
    data["y"] = rng.negative_binomial(2, 2 / (2 + mu))
    train, test = train_test_split(data, test_size=0.25, random_state=106)
    model = GAM(
        formula="y ~ s(x,bs='cr',k=12) + offset(log_exposure)",
        family=NegativeBinomialLogFamily(theta=1, estimate_theta=True),
        optimize_smoothing=True,
        smoothing_method="reml",
    ).fit(train)
    assert np.isfinite(model.family.getTheta(True)).all()
    assert np.all(np.asarray(model.family.getTheta(True)) > 0)
    assert np.isfinite(model.predict(test)).all()
    assert np.isfinite(model.smoothing_params).all()
