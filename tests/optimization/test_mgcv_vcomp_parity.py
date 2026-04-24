"""Focused parity checks for ``gam_vcomp()`` against mgcv."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from nampy.gam.smoothing_selection import postfit as postfit_module
from tests.families.test_general_family_mgcv_parity import (
    _gammals_data,
    _gaulss_data,
    _gevlss_data,
    _shashlss_data,
    _ziplss_data,
)
from tests.mgcv_parity_utils import (
    _fit_nampy_model,
    _make_gaussian_data,
    _make_negbin_data,
    _make_poisson_data,
    _run_mgcv_gam_vcomp,
)

pytestmark = [pytest.mark.surface_output, pytest.mark.surface_regression]


def _as_float_array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _assert_gam_vcomp_close(actual, expected, *, atol: float) -> None:
    assert actual is not None
    assert expected is not None
    assert actual.get("names", None) == expected.get("names", None)
    np.testing.assert_allclose(
        _as_float_array(actual["vc"]),
        _as_float_array(expected["vc"]),
        atol=atol,
        rtol=atol,
    )

    expected_all = expected.get("all", None)
    actual_all = actual.get("all", None)
    if expected_all is None:
        assert actual_all is None
    else:
        assert actual.get("all_names", None) == expected.get("all_names", None)
        np.testing.assert_allclose(
            _as_float_array(actual_all),
            _as_float_array(expected_all),
            atol=atol,
            rtol=atol,
        )

    assert actual.get("rank", None) == expected.get("rank", None)
    assert actual.get("rank_hess", None) == expected.get("rank_hess", None)
    if expected.get("conf_lev", None) is None:
        assert actual.get("conf_lev", None) is None
    else:
        np.testing.assert_allclose(
            float(actual["conf_lev"]),
            float(expected["conf_lev"]),
            atol=0.0,
            rtol=0.0,
        )


def _fake_gam_vcomp_model(*, compiled_penalties, n_smoothing_params: int):
    return SimpleNamespace(
        _fitted=True,
        smoothing_params=np.ones(n_smoothing_params, dtype=np.float64),
        compiled_model_=SimpleNamespace(
            compiled_penalties=tuple(compiled_penalties),
            n_smoothing_params=n_smoothing_params,
        ),
    )


@pytest.mark.parametrize(
    ("compiled_penalties", "n_smoothing_params", "expected_message"),
    [
        (
            (
                SimpleNamespace(
                    smoothing_index=0,
                    smoothing_id="smooth_term",
                    metadata={},
                    is_null_space_penalty=False,
                ),
            ),
            1,
            "gam_vcomp(rescale=True) missing exact mgcv penalty rescale metadata "
            "for smooth_term.",
        ),
        (
            (
                SimpleNamespace(
                    smoothing_index=0,
                    metadata={"mgcv_s_scale": 2.0},
                    is_null_space_penalty=False,
                ),
                SimpleNamespace(
                    smoothing_index=0,
                    metadata={"mgcv_s_scale": 3.0},
                    is_null_space_penalty=False,
                ),
            ),
            1,
            "gam_vcomp(rescale=True) requires one exact mgcv penalty rescale "
            "factor per smoothing parameter; index 0 has 2.0 and 3.0.",
        ),
        (
            (
                SimpleNamespace(
                    smoothing_index=0,
                    metadata={"mgcv_s_scale": 2.0},
                    is_null_space_penalty=False,
                ),
            ),
            2,
            "gam_vcomp(rescale=True) missing penalty metadata for smoothing "
            "parameter indices 1.",
        ),
    ],
)
def test_gam_vcomp_rescale_true_metadata_errors(
    compiled_penalties, n_smoothing_params, expected_message
):
    """Verify that gam vcomp rescale true metadata errors."""
    model = _fake_gam_vcomp_model(
        compiled_penalties=compiled_penalties,
        n_smoothing_params=n_smoothing_params,
    )

    with pytest.raises(NotImplementedError) as excinfo:
        postfit_module.gam_vcomp(model, rescale=True)

    assert str(excinfo.value) == expected_message


def test_gam_vcomp_rescale_true_matches_mgcv_gcv():
    """Verify that gam vcomp rescale true matches mgcv GCV."""
    data = _make_gaussian_data(seed=41, n=120)
    formula = 'y ~ s(x0, bs="cr", k=8)'

    expected = _run_mgcv_gam_vcomp(
        data,
        formula,
        "gaussian",
        "GCV",
        rescale=True,
    )
    gam = _fit_nampy_model(data, formula, "gaussian", "GCV")

    actual = gam.gam_vcomp(rescale=True)

    np.testing.assert_allclose(
        _as_float_array(actual["vc"]),
        _as_float_array(expected["vc"]),
        atol=5e-8,
        rtol=0.0,
    )


def test_gam_vcomp_rescale_true_matches_mgcv_reml_ci():
    """Verify that gam vcomp rescale true matches mgcv REML ci."""
    data = _make_gaussian_data(seed=42, n=140)
    formula = 'y ~ s(x0, bs="cr", k=8)'

    expected = _run_mgcv_gam_vcomp(
        data,
        formula,
        "gaussian",
        "REML",
        rescale=True,
    )
    gam = _fit_nampy_model(data, formula, "gaussian", "REML")

    actual = gam.gam_vcomp(rescale=True)

    np.testing.assert_allclose(
        _as_float_array(actual["vc"]),
        _as_float_array(expected["vc"]),
        atol=1e-4,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("data_factory", "formula", "family", "method", "rescale", "atol"),
    [
        (
            lambda: _make_gaussian_data(seed=41, n=120),
            'y ~ s(x0, bs="cr", k=8)',
            "gaussian",
            "GCV",
            False,
            5e-8,
        ),
        (
            lambda: _make_poisson_data(seed=789, n=140),
            'y ~ s(x0, bs="cr", k=8)',
            "poisson",
            "REML",
            False,
            2e-5,
        ),
        (
            lambda: _make_negbin_data(seed=77, n=140),
            'y ~ s(x0, bs="cr", k=8)',
            {"name": "negbin", "theta": 2.5, "estimate_theta": True},
            "REML",
            False,
            5e-5,
        ),
        (
            _gaulss_data,
            ['y ~ s(x, bs="cr", k=6)', "~ 1"],
            "gaulss",
            "ML",
            False,
            2e-5,
        ),
        (
            _gammals_data,
            ['y ~ s(x, bs="cr", k=6)', "~ 1"],
            "gammals",
            "ML",
            False,
            2e-5,
        ),
        (
            _gevlss_data,
            ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1"],
            "gevlss",
            "ML",
            False,
            3e-5,
        ),
        (
            _shashlss_data,
            ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
            "shashlss",
            "ML",
            False,
            8e-5,
        ),
        (
            _ziplss_data,
            ['y ~ s(x, bs="cr", k=6)', "~ 1"],
            "ziplss",
            "ML",
            False,
            5e-5,
        ),
    ],
    ids=[
        "gaussian_gcv_rescale_false",
        "poisson_reml_rescale_false",
        "negbin_est_reml_rescale_false",
        "gaulss_ml_rescale_false",
        "gammals_ml_rescale_false",
        "gevlss_ml_rescale_false",
        "shashlss_ml_rescale_false",
        "ziplss_ml_rescale_false",
    ],
)
def test_gam_vcomp_matches_mgcv_requested_surface(
    data_factory, formula, family, method, rescale, atol
):
    """Verify that gam vcomp matches mgcv requested surface."""
    data = data_factory()
    expected = _run_mgcv_gam_vcomp(
        data,
        formula,
        family,
        method,
        rescale=rescale,
    )
    gam = _fit_nampy_model(data, formula, family, method)

    actual = gam.gam_vcomp(rescale=rescale)

    _assert_gam_vcomp_close(actual, expected, atol=atol)
