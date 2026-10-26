"""Source penalty-root evaluation at the flat NB smoothing boundary."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nampy.gam import GAM
from nampy.gam.fit.backends import solve_pirls_given_smoothing
from nampy.gam.fit.selection.criteria.pirls.derivatives import _gdi1_kernel
from nampy.gam.fit.selection.criteria.pirls.value import (
    _pirls_ml_reml_objective_from_solution,
)

ROOT = Path(__file__).parents[1] / "reference_fixtures/mgcv_negbin_boundary"
LOG_THETA = -0.12400663050520905
LOG_SP = 20.20394495173847


def test_negbin_boundary_criterion_uses_source_penalty_roots():
    data = pd.DataFrame(json.loads((ROOT / "inputs.json").read_text()))
    expected = json.loads((ROOT / "reference.json").read_text())["captured"]
    sp = np.exp([LOG_SP])
    model = GAM(
        formula='y ~ s(x0, bs="cr", k=8)',
        family={"name": "negbin", "theta": np.exp(LOG_THETA)},
        smoothing_params=sp,
        optimize_smoothing=False,
    ).fit(data=data)
    y = data["y"].to_numpy()
    solution = solve_pirls_given_smoothing(model, y, sp)
    kernel = _gdi1_kernel(model, y, solution, sp, method="REML")
    actual = _pirls_ml_reml_objective_from_solution(model, y, solution, sp, "REML")
    np.testing.assert_allclose(kernel.bSb, expected["penalty"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual, expected["criterion"], rtol=0, atol=1e-10)
