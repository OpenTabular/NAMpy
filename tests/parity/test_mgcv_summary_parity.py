"""Direct summary.gam parity against mgcv (mgcv/R/mgcv.r:3858-4068)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.inference.summary import summary_gam
from tests.families.test_general_family_mgcv_parity import (
    GAULSS_FORMULA,
    _gammals_data,
    _gaulss_data,
)
from tests.mgcv_parity_utils import (
    _make_binomial_data,
    _make_gamma_data,
    _make_gaussian_data,
    _make_negbin_data,
    _make_poisson_data,
    _run_mgcv_snapshot,
)

pytestmark = [pytest.mark.surface_output]


def _gaussian_offset_data(seed=19, n=110):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(size=n)
    off = rng.uniform(-0.3, 0.3, size=n)
    data = pd.DataFrame({"x0": x0, "off": off})
    data["y"] = np.sin(2.0 * np.pi * x0) + off + 0.15 * rng.standard_normal(n)
    return data


_CASES = [
    (
        "gaussian_reml",
        "gaussian",
        'y ~ x0 + s(x1, bs="cr", k=8)',
        lambda: _make_gaussian_data(seed=417, n=160),
        "REML",
    ),
    (
        "gaussian_offset_reml",
        "gaussian",
        'y ~ offset(off) + s(x0, bs="cr", k=8)',
        _gaussian_offset_data,
        "REML",
    ),
    (
        "poisson_reml",
        "poisson",
        'y ~ s(x0, bs="cr", k=8)',
        _make_poisson_data,
        "REML",
    ),
    (
        "gamma_joint_scale_reml",
        "gamma",
        'y ~ s(x0, bs="cr", k=8)',
        _make_gamma_data,
        "REML",
    ),
    (
        "negbin_est_theta_reml",
        {"name": "negbin", "theta": 1.8, "estimate_theta": True},
        'y ~ s(x0, bs="cr", k=8)',
        _make_negbin_data,
        "REML",
    ),
    (
        "gaulss_ml",
        "gaulss",
        GAULSS_FORMULA,
        _gaulss_data,
        "ML",
    ),
    (
        "binomial_reml",
        "binomial",
        'y ~ x1 + s(x0, bs="cr", k=8)',
        _make_binomial_data,
        "REML",
    ),
    (
        "negbin_fixed_theta_reml",
        {"name": "negbin", "theta": 1.6},
        'y ~ s(x0, bs="cr", k=8)',
        _make_negbin_data,
        "REML",
    ),
    (
        "gaussian_gcv",
        "gaussian",
        'y ~ x0 + s(x1, bs="cr", k=8)',
        lambda: _make_gaussian_data(seed=417, n=160),
        "GCV",
    ),
    (
        # Two smooths: the single-smooth UBRE optimum is flat enough that the
        # optimizer endpoints differ at ~1e-5, moving p-table estimates past
        # the strict tolerance; the two-smooth endpoint matches mgcv strictly
        # (see test_mgcv_criterion_optimizer_endpoint_parity.py).
        "poisson_ubre",
        "poisson",
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        _make_poisson_data,
        "UBRE",
    ),
    (
        "gaussian_no_intercept_reml",
        "gaussian",
        'y ~ x0 + s(x1, bs="cr", k=8) - 1',
        lambda: _make_gaussian_data(seed=419, n=150),
        "REML",
    ),
    (
        "gammals_ml",
        "gammals",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _gammals_data,
        "ML",
    ),
]

_MGCV_METHOD_TOKENS = {"GCV": "GCV.Cp", "UBRE": "GCV.Cp"}


def _normalized_smooth_label(label: str) -> str:
    """mgcv prints s(x); nampy keeps the full term text with bs=/k= args."""
    text = str(label)
    if "(" not in text:
        return text
    head, args = text.split("(", 1)
    first_arg = args.rstrip(")").split(",")[0].strip()
    return f"{head}({first_arg})"


def _label_list(value) -> list[str]:
    """jsonlite auto_unbox collapses single-label vectors to a bare string."""
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _opt(value):
    """R NULL serializes as {} / [] through jsonlite; normalize to None."""
    if value is None or (isinstance(value, (dict, list)) and len(value) == 0):
        return None
    return value


def _assert_p_value_columns_close(actual, expected, *, atol=1e-6):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape
    both = np.isfinite(actual) & np.isfinite(expected)
    assert np.array_equal(np.isfinite(actual), np.isfinite(expected))
    np.testing.assert_allclose(actual[both], expected[both], rtol=1e-3, atol=atol)


@pytest.mark.parametrize(
    ("case_id", "family", "formula", "data_factory", "method"),
    _CASES,
    ids=[case[0] for case in _CASES],
)
def test_summary_gam_matches_mgcv(case_id, family, formula, data_factory, method):
    data = data_factory()
    gam = GAM(
        family=family,
        formula=formula,
        optimize_smoothing=True,
        smoothing_method=method,
    )
    gam.fit(data=data)
    summary = summary_gam(gam)

    mgcv_method = _MGCV_METHOD_TOKENS.get(method, method)
    expected = _run_mgcv_snapshot(data, formula, family, mgcv_method)
    block = expected["parity"]["diagnostics"]["summary"]
    assert block is not None

    exp_pt = np.asarray(block["p_table"]["values"], dtype=np.float64)
    act_pt = summary.p_table.to_numpy(dtype=np.float64)
    assert act_pt.shape == exp_pt.shape
    np.testing.assert_allclose(act_pt[:, 0], exp_pt[:, 0], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(act_pt[:, 1], exp_pt[:, 1], rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(act_pt[:, 2], exp_pt[:, 2], rtol=1e-5, atol=1e-8)
    _assert_p_value_columns_close(act_pt[:, 3], exp_pt[:, 3])
    exp_cols = [str(c) for c in block["p_table"]["columns"]]
    assert list(summary.p_table.columns) == exp_cols

    assert summary.residual_df == pytest.approx(float(block["residual_df"]), rel=1e-6)
    assert summary.scale == pytest.approx(float(block["scale"]), rel=1e-6)
    assert summary.n == int(block["n"])
    assert summary.np == int(block["np"])
    assert summary.method == str(block["method"])
    sp_criterion = _opt(block.get("sp_criterion"))
    if sp_criterion is None:
        assert summary.sp_criterion is None
    else:
        assert summary.sp_criterion == pytest.approx(
            float(np.ravel(sp_criterion)[0]), rel=1e-6
        )

    r_sq = _opt(block.get("r_sq"))
    if r_sq is None:
        assert summary.r_sq is None
    else:
        assert summary.r_sq == pytest.approx(float(np.ravel(r_sq)[0]), rel=1e-5)
    dev_expl = _opt(block.get("dev_expl"))
    if dev_expl is None:
        assert summary.dev_expl is None
    else:
        assert summary.dev_expl == pytest.approx(float(np.ravel(dev_expl)[0]), rel=1e-5)
    null_dev = _opt(expected["fit"].get("null_deviance"))
    if null_dev is not None:
        assert summary.null_deviance == pytest.approx(
            float(np.ravel(null_dev)[0]), rel=1e-5
        )

    # Full smooth table (summary.gam s.table): labels, edf/ref_df, statistic
    # and p-value.  The snapshot's anova_smooth block IS the summary s.table
    # (anova.gam single-model is summary.gam, mgcv/R/mgcv.r:4153).
    exp_smooth = _opt(expected["parity"]["diagnostics"].get("anova_smooth"))
    if exp_smooth is None:
        assert len(summary.s_table) == 0
    else:
        exp_labels = [
            _normalized_smooth_label(v) for v in _label_list(exp_smooth["labels"])
        ]
        act_labels = [_normalized_smooth_label(v) for v in summary.s_table["label"]]
        assert act_labels == exp_labels
        exp_values = np.atleast_2d(np.asarray(exp_smooth["values"], dtype=np.float64))
        act_values = summary.s_table[
            ["edf", "ref_df", "wald_stat", "p_value"]
        ].to_numpy(dtype=np.float64)
        assert act_values.shape == exp_values.shape
        # General-family optimizer endpoints vary at the low-ppm level across
        # supported LAPACK wheels; the table still derives from the same fit.
        np.testing.assert_allclose(
            act_values[:, :2], exp_values[:, :2], rtol=2e-6, atol=1e-8
        )
        np.testing.assert_allclose(
            act_values[:, 2], exp_values[:, 2], rtol=1e-5, atol=1e-8
        )
        # Near-zero smooth p-values wobble at the endpoint-precision level.
        _assert_p_value_columns_close(act_values[:, 3], exp_values[:, 3], atol=1e-5)

    # Parametric-term Wald table (summary.gam pTerms.table) where present.
    exp_pterms = _opt(expected["parity"]["diagnostics"].get("anova_parametric"))
    if exp_pterms is None:
        assert len(summary.pterms_table) == 0
    else:
        assert _label_list(exp_pterms["labels"]) == [
            str(v) for v in summary.pterms_table["label"]
        ]
        exp_values = np.atleast_2d(np.asarray(exp_pterms["values"], dtype=np.float64))
        act_values = summary.pterms_table[["df", "wald_stat", "p_value"]].to_numpy(
            dtype=np.float64
        )
        assert act_values.shape == exp_values.shape
        np.testing.assert_allclose(
            act_values[:, 0], exp_values[:, 0], rtol=0.0, atol=1e-8
        )
        np.testing.assert_allclose(
            act_values[:, 1], exp_values[:, 1], rtol=1e-5, atol=1e-8
        )
        _assert_p_value_columns_close(act_values[:, 2], exp_values[:, 2])
