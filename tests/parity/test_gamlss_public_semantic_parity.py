"""Cross-package parity for the public GAMLSS natural-parameter contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from nampy import GAMLSS

_FIXTURE_DIR = Path(__file__).parents[1] / "reference_fixtures" / "gamlss"


def _reference_data(family: str) -> pd.DataFrame:
    n = 80
    x = np.linspace(-1.5, 1.5, n)
    phase = np.arange(1, n + 1)
    z = np.sin(phase * 1.7) + 0.5 * np.cos(phase * 0.41)
    if family == "normal":
        y = 0.3 + 0.9 * x + np.exp(-0.4 + 0.2 * x) * z
    elif family == "gamma":
        y = np.exp(0.2 + 0.5 * x) * np.exp(0.3 * z)
    else:  # pragma: no cover - test parameter guard
        raise ValueError(family)
    return pd.DataFrame({"x": x, "y": y})


@pytest.mark.parametrize(
    ("family", "gamlss_parameter_atol", "gamlss_density_atol"),
    [("normal", 5e-4, 1e-3), ("gamma", 1e-5, 5e-5)],
)
def test_parametric_natural_parameters_match_mgcv_and_r_gamlss(
    family, gamlss_parameter_atol, gamlss_density_atol
):
    """mgcv must match numerically; R gamlss must match public semantics.

    The two R packages use different fitting algorithms and link details, so
    their small optimizer-level differences are deliberately not treated as
    evidence against semantic parity.
    """
    data = _reference_data(family)
    reference = pd.read_csv(_FIXTURE_DIR / f"{family}.csv")
    estimator = GAMLSS(
        family=family,
        formula={"mu": "y ~ x", "sigma": "~ x"},
        optimize_smoothing=False,
    ).fit(data)

    rows = reference["row"].to_numpy(dtype=int) - 1
    parameters = estimator.predict(data)[rows]
    mgcv_parameters = reference[["mgcv_mu", "mgcv_sigma"]].to_numpy()
    gamlss_parameters = reference[["gamlss_mu", "gamlss_sigma"]].to_numpy()

    assert_allclose(parameters, mgcv_parameters, rtol=1e-11, atol=1e-11)
    assert_allclose(
        parameters,
        gamlss_parameters,
        rtol=0.0,
        atol=gamlss_parameter_atol,
    )

    logpdf = estimator.gam_.family.logpdf_from_parameters(
        data["y"].to_numpy()[rows], parameters
    )
    assert_allclose(
        logpdf,
        reference["gamlss_logpdf"].to_numpy(),
        rtol=0.0,
        atol=gamlss_density_atol,
    )
