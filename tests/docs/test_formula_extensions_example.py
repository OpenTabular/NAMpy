"""Runnable examples for the supported R-style RHS formula transformations."""

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ factor(group) + x",
        "y ~ ordered(group) + x",
        "y ~ interaction(group,site) + x",
        "y ~ cut(x,breaks=[-3,0,3],labels=['low','high']) + group",
        "y ~ poly(x,degree=3)",
    ],
)
def test_formula_extension_example(formula):
    rng = np.random.default_rng(77)
    x = np.linspace(-2, 2, 120)
    data = pd.DataFrame(
        {
            "x": x,
            "group": np.resize([1, 2, 3], 120),
            "site": np.resize(["east", "west"], 120),
        }
    )
    data["y"] = 0.5 + x - 0.2 * x * x + 0.1 * data.group + rng.normal(0, 0.2, 120)
    model = GAM(formula=formula, optimize_smoothing=False).fit(data=data)
    new_data = pd.DataFrame(
        {"x": [-0.5, 0.3], "group": [1, 3], "site": ["east", "west"]}
    )
    prediction, standard_error = model.predict(new_data, return_se=True)
    assert prediction.shape == standard_error.shape == (2,)
    assert np.isfinite(prediction).all()
    assert (standard_error > 0).all()
