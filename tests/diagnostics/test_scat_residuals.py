"""SCAT stored working residuals follow gam.fit4's offset convention."""

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.families.scat import ScaledTFamily
from tests.families.test_scat_family_contracts import reference


@pytest.mark.parametrize("zero", [False, True])
def test_scat_working_residuals_offsets_and_zero_weights(zero):
    x = np.linspace(-1, 1, 25)
    offset = 0.1 * np.cos(x)
    y = 0.2 * np.sin(2 * x) + offset + 0.03 * np.cos(17 * x)
    weights = np.ones_like(x)
    if zero:
        weights[[0, 12]] = 0
    data = {
        "x": x.tolist(),
        "y": y.tolist(),
        "off": offset.tolist(),
        "w": weights.tolist(),
    }
    code = """
d<-as.data.frame(a)
b<-gam(y~s(x,bs='cr',k=5)+offset(off),data=d,weights=d$w,family=scat(theta=c(6,1)),sp=.4,method='REML')
ans<-list(working=residuals(b,type='working'),fitted=fitted(b))
"""
    ref = reference(data, code)
    model = GAM(
        formula="y~s(x,bs='cr',k=5)+offset(off)",
        family=ScaledTFamily(theta=[6, 1]),
        smoothing_params=[0.4],
        optimize_smoothing=False,
    ).fit(data=pd.DataFrame(data), sample_weight="w")
    actual = model.residuals(type="working")
    expected = np.asarray(
        [np.nan if value == "NA" else value for value in ref["working"]], dtype=float
    )
    np.testing.assert_allclose(model.predict(), ref["fitted"], atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-8)
    np.testing.assert_array_equal(np.isnan(actual), weights == 0)
