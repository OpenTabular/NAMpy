"""Public family forms, persistence, and explicitly classified boundaries."""

import numpy as np
import pytest

from nampy.gam import GAM
from nampy.gam.families import CensoredLogisticFamily
from nampy.models import GAMRegressor
from tests.families.test_mgcv_cnorm_workflows import _case, _data, _newdata


@pytest.mark.parametrize("spec", ["clog", {"name": "clog"}, ("clog", {}), CensoredLogisticFamily()])
def test_clog_estimator_forms_and_persistence(spec, tmp_path):
    data = _data(_case("mixed"))
    model = GAMRegressor(formula="cbind(y,bound)~s(x,bs='cr',k=7)", family=spec,
                         optimize_smoothing=True, smoothing_method="reml").fit(data)
    components = model.predict_components(_newdata())
    components.validate_additive_reconstruction()
    path = tmp_path / "clog.pkl"
    model.save_model(path)
    restored = GAMRegressor.load_model(path)
    np.testing.assert_array_equal(restored.predict(_newdata()), model.predict(_newdata()))
    np.testing.assert_array_equal(restored.gam_.response_.bounds, model.gam_.response_.bounds)
    assert not restored.gam_.response_.bounds.flags.writeable


def test_clog_zero_precision_is_an_acceptance_boundary():
    case = _case("mixed")
    data = _data(case)
    weights = np.ones(len(data))
    weights[0] = 0
    with pytest.raises(NotImplementedError, match="positive precision"):
        GAM(formula=case["formula"], family=CensoredLogisticFamily(theta=1),
            optimize_smoothing=False, smoothing_params=[0.7]).fit(data, sample_weight=weights)


def test_clog_efs_estimation_has_public_component_contract(tmp_path):
    case = _case("mixed")
    model = GAMRegressor(formula=case["formula"], family="clog", smoothing_method="reml",
            smoothing_optimizer="efs").fit(_data(case))
    model.predict_components(_newdata()).validate_additive_reconstruction()
    assert model.gam_.family.n_theta == 1
    path = tmp_path / "clog-efs.pkl"
    model.save_model(path)
    restored = GAMRegressor.load_model(path)
    np.testing.assert_array_equal(restored.predict(_newdata()), model.predict(_newdata()))
    np.testing.assert_array_equal(restored.gam_.family.getTheta(), model.gam_.family.getTheta())
    assert not restored.gam_.response_.bounds.flags.writeable
