"""Shared estimator contracts for explicitly censored observations."""

import numpy as np

from nampy.gam.families.cnorm import CensoredNormalFamily
from nampy.models.gam import GAMRegressor
from tests.families.test_mgcv_cnorm_workflows import _case, _data, _reference


def test_cnorm_adapter_prediction_components_match_upstream():
    case = _case("mixed", offset=True)
    reference = _reference(case)
    model = GAMRegressor(
        formula=case["formula"],
        family=CensoredNormalFamily(theta=1),
        smoothing_params=[0.7],
        optimize_smoothing=False,
    ).fit(_data(case))
    import pandas as pd

    newdata = pd.DataFrame(
        {"x": np.linspace(-0.93, 0.93, 19), "off": np.linspace(-0.08, 0.08, 19)}
    )
    np.testing.assert_allclose(
        model.predict(newdata), reference["prediction"], atol=2e-7, rtol=2e-7
    )
    components = model.predict_components(newdata)
    components.validate_additive_reconstruction()
    np.testing.assert_allclose(
        components.link[:, 0], reference["prediction"], atol=2e-7, rtol=2e-7
    )


def test_cnorm_zero_precision_fitting_is_explicitly_guarded():
    import pytest

    from nampy.gam import GAM

    case = _case("mixed")
    data = _data(case)
    weights = np.ones(len(data))
    weights[0] = 0
    with pytest.raises(NotImplementedError, match="positive precision weights"):
        GAM(
            formula=case["formula"],
            family=CensoredNormalFamily(theta=1),
            optimize_smoothing=False,
            smoothing_params=[0.7],
        ).fit(data, sample_weight=weights)


def test_cnorm_positive_min_sp_is_explicitly_guarded():
    import pytest

    from nampy.gam import GAM

    case = _case("mixed")
    with pytest.raises(NotImplementedError, match="positive min_sp"):
        GAM(
            formula=case["formula"],
            family=CensoredNormalFamily(theta=1),
            optimize_smoothing=False,
            smoothing_params=[0.7],
            min_sp=[0.01],
        ).fit(_data(case))


def test_cnorm_public_family_forms_match_mgcv():
    from nampy.gam import GAM
    from tests.families.test_mgcv_cnorm_workflows import (
        _assert_fit,
        _selection_case,
        _selection_reference,
    )

    fixed_case = _case("mixed")
    for spec in ({"name": "cnorm", "theta": 1}, ("cnorm", {"theta": 1})):
        fit = GAM(
            formula=fixed_case["formula"],
            family=spec,
            smoothing_params=[0.7],
            optimize_smoothing=False,
        ).fit(_data(fixed_case))
        _assert_fit(fixed_case, _reference(fixed_case), fit)
    case = _selection_case("mixed", method="REML")
    fit = GAM(
        formula=case["formula"],
        family="cnorm",
        optimize_smoothing=True,
        smoothing_method="reml",
    ).fit(_data(case))
    _assert_fit(case, _selection_reference(case), fit)


def test_cnorm_estimated_efs_is_explicitly_guarded():
    import pytest

    from nampy.gam import GAM

    case = _case("mixed")
    with pytest.raises(NotImplementedError, match="EFS"):
        GAM(
            formula=case["formula"],
            family="cnorm",
            smoothing_method="reml",
            smoothing_optimizer="efs",
            optimize_smoothing=True,
        ).fit(_data(case))


def test_cnorm_unsupported_fixed_sp_ancillary_criteria_raise():
    import pytest

    from nampy.gam import GAM

    case = _case("mixed")
    for method in ("NCV", "not-a-method"):
        with pytest.raises((NotImplementedError, ValueError)):
            GAM(
                formula=case["formula"],
                family="cnorm",
                smoothing_params=[0.7],
                smoothing_method=method,
                optimize_smoothing=False,
            ).fit(_data(case))
