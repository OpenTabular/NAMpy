"""Explicit boundaries of the directly validated scaled-t workflow."""

import pytest

from tests.families.test_mgcv_scat_workflows import _case, _fit


@pytest.mark.parametrize("link", ["identity", "log", "inverse"])
@pytest.mark.parametrize("estimated", [False, True])
def test_scat_ml_zero_weights_are_explicitly_guarded(link, estimated):
    # Nine committed actual-R probes record the upstream segmentation fault.
    with pytest.raises(NotImplementedError, match="mgcv 1.9-4.*crashes"):
        _fit(_case(link=link, estimated=estimated, method="ML", zero=True))


def test_scat_estimated_theta_efs_is_explicitly_guarded():
    import pandas as pd

    from nampy.gam import GAM

    case = _case(estimated=True)
    with pytest.raises(NotImplementedError, match="vector-theta.*EFS"):
        GAM(family="scat", formula=case["formula"], smoothing_method="REML",
            smoothing_optimizer="efs").fit(data=pd.DataFrame(case["data"]))
