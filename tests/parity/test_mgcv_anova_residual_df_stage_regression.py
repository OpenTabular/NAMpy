from __future__ import annotations

import numpy as np
import pytest

from nampy.gam._model_state import (
    _coef_column_offset,
    _edf2,
    _edf_total,
    _summary_R,
    _term_blocks_seq,
)
from nampy.gam.inference.anova import _edf1_vector, _residual_df_approx_mgcv
from tests.mgcv_parity_utils import _fit_nampy_model, _make_gamma_data, _run_mgcv_snapshot

pytestmark = [pytest.mark.surface_output, pytest.mark.surface_regression]


_GAMMA_COMPARISON_FORMULAS = [
    pytest.param('y ~ s(x0, bs="cr", k=8)', id="gamma_one_cr"),
    pytest.param(
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        id="gamma_two_cr",
    ),
]


@pytest.mark.parametrize("formula", _GAMMA_COMPARISON_FORMULAS)
def test_gamma_anova_residual_df_stage_matches_mgcv_snapshot_components(formula: str):
    """Verify that gamma anova residual-df ingredients match mgcv snapshot components."""
    data = _make_gamma_data()
    expected = _run_mgcv_snapshot(data, formula, "gamma", "REML")
    gam = _fit_nampy_model(data, formula, "gamma", "REML")

    x_off = _coef_column_offset(gam)
    edf1_vec = np.asarray(_edf1_vector(gam), dtype=np.float64)
    actual_smooth_edf1 = []
    for tb in _term_blocks_seq(gam):
        if str(getattr(tb, "term_type", "")) == "parametric":
            continue
        sl = slice(
            int(tb.coef_slice.start) + x_off,
            int(tb.coef_slice.stop) + x_off,
        )
        actual_smooth_edf1.append(float(np.sum(edf1_vec[sl])))

    expected_smooth_edf1 = np.asarray(
        expected["parity"]["diagnostics"]["smooth_edf1"]["values"],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        np.asarray(actual_smooth_edf1, dtype=np.float64),
        expected_smooth_edf1,
        atol=5e-6,
        rtol=5e-6,
    )

    expected_r_blocks = expected["parity"]["diagnostics"]["smooth_test_inputs"]["r_blocks"]
    assert expected_r_blocks is not None
    R_full = np.asarray(_summary_R(gam), dtype=np.float64)
    actual_r_blocks = []
    for tb in _term_blocks_seq(gam):
        if str(getattr(tb, "term_type", "")) == "parametric":
            continue
        actual_r_blocks.append(
            np.asarray(
                R_full[
                    :,
                    int(tb.coef_slice.start) + x_off : int(tb.coef_slice.stop) + x_off,
                ],
                dtype=np.float64,
            )
        )
    for actual_block, expected_block in zip(actual_r_blocks, expected_r_blocks):
        np.testing.assert_allclose(
            actual_block,
            np.asarray(expected_block, dtype=np.float64),
            atol=5e-6,
            rtol=5e-6,
        )

    intercept_edf1 = float(np.sum(edf1_vec[:x_off])) if x_off else 0.0
    np.testing.assert_allclose(
        np.asarray([intercept_edf1], dtype=np.float64),
        np.asarray([float(x_off)], dtype=np.float64),
        atol=5e-6,
        rtol=5e-6,
    )

    actual_edf2 = _edf2(gam)
    expected_edf2 = expected["fit"]["edf2"]
    assert actual_edf2 is not None
    assert expected_edf2 is not None
    np.testing.assert_allclose(
        np.asarray(actual_edf2, dtype=np.float64),
        np.asarray(expected_edf2, dtype=np.float64),
        atol=5e-6,
        rtol=5e-6,
    )

    expected_resid_df = float(data.shape[0]) - (
        float(x_off)
        + float(np.sum(expected_smooth_edf1))
        + float(np.sum(np.asarray(expected_edf2, dtype=np.float64)))
        - float(expected["fit"]["edf_total"])
    )
    np.testing.assert_allclose(
        np.asarray([_residual_df_approx_mgcv(gam)], dtype=np.float64),
        np.asarray([expected_resid_df], dtype=np.float64),
        atol=5e-6,
        rtol=5e-6,
    )

    # Keep the derived snapshot-side residual-df decomposition consistent with
    # the model-comparison anova surface that ultimately consumes it.
    assert float(_edf_total(gam)) > 0.0
