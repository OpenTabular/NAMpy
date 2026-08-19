"""from_fitted_gam fidelity: compiled design equals the GAM lpmatrix.

``CompiledGAMTerms.design(X_new)`` and ``GAM.lpmatrix(X_new)`` must agree
column-for-column (the lpmatrix prepends the intercept column) — the bridge
consumes exactly the mgcv-parity prediction basis, nothing else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.hybrid import CompiledGAMTerms


def _frame(n=150, seed=0):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {
            "x0": rng.uniform(size=n),
            "x1": rng.uniform(size=n),
            "z": rng.normal(size=n),
        }
    )
    data["y"] = (
        np.sin(3.0 * data["x0"])
        + data["x1"] ** 2
        + 0.5 * data["z"]
        + rng.normal(scale=0.1, size=n)
    )
    return data


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ s(x0, k=6)",
        "y ~ s(x0, k=6, bs='cr')",
        "y ~ s(x0, k=6) + s(x1, k=5) + z",
    ],
    ids=["tp_smooth", "cr_smooth", "multi_smooth_plus_linear"],
)
def test_design_matches_lpmatrix_without_intercept_column(formula):
    data = _frame()
    new_rows = data.iloc[:40]

    gam = GAM(
        formula=formula,
        family="gaussian",
        optimize_smoothing=True,
        smoothing_method="reml",
    )
    gam.fit(data=data)

    terms = CompiledGAMTerms.from_fitted_gam(gam)
    design = terms.design(new_rows)
    lpmatrix = gam.lpmatrix(new_rows)

    assert design.shape == (len(new_rows), lpmatrix.shape[1] - 1)
    np.testing.assert_allclose(lpmatrix[:, 0], 1.0, atol=0.0)
    np.testing.assert_allclose(design, lpmatrix[:, 1:], atol=1e-12)
