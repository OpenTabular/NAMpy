"""Public ziP routing and explicit boundaries for unaccepted combinations."""

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.families import ZeroInflatedPoissonFamily
from tests.families.test_mgcv_zip_workflows import _assert_fit, _case, _reference


@pytest.mark.parametrize(
    "spec",
    [
        {"name": "ziP", "theta": [-0.5, np.log(0.5)], "b": 0.3},
        ("ziP", {"theta": [-0.5, np.log(0.5)], "b": 0.3}),
        ZeroInflatedPoissonFamily(theta=[-0.5, np.log(0.5)], b=0.3),
    ],
)
def test_zip_public_specification_matches_upstream(spec):
    case = _case()
    model = GAM(
        family=spec, formula=case["formula"], smoothing_params=case["sp"],
        optimize_smoothing=False,
    ).fit(data=pd.DataFrame(case["data"]))
    _assert_fit(case, _reference(case), model)


@pytest.mark.parametrize("estimated", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("weights", ["nonunit", "zero"])
def test_zip_nonunit_weights_are_explicitly_guarded(estimated, optimize, weights):
    case = _case(estimated=estimated, weights=weights)
    with pytest.raises(NotImplementedError, match="unit prior weights only"):
        GAM(
            family=ZeroInflatedPoissonFamily(theta=case["theta"], b=0.3),
            formula=case["formula"], optimize_smoothing=optimize,
            smoothing_params=[0.4], smoothing_method="REML",
        ).fit(data=pd.DataFrame(case["data"]), sample_weight="w")


@pytest.mark.parametrize("estimated", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
def test_zip_positive_min_sp_is_explicitly_guarded(estimated, optimize):
    case = _case(estimated=estimated)
    with pytest.raises(NotImplementedError, match="positive min_sp"):
        GAM(
            family=ZeroInflatedPoissonFamily(theta=case["theta"], b=0.3),
            formula=case["formula"], optimize_smoothing=optimize,
            smoothing_params=[0.4], min_sp=[0.1],
        ).fit(data=pd.DataFrame(case["data"]))


def test_zip_estimated_theta_efs_is_explicitly_guarded():
    case = _case(estimated=True)
    with pytest.raises(NotImplementedError, match="vector-theta.*EFS"):
        GAM(
            family="ziP", formula=case["formula"],
            optimize_smoothing=True, smoothing_optimizer="efs",
            smoothing_method="REML",
        ).fit(data=pd.DataFrame(case["data"]))
