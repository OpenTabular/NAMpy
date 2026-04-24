from __future__ import annotations

import numpy as np
import pytest

from tests.optimization._trace_parity_helpers import (
    LINKED_ID_TRACE_CASES,
    _assert_mgcv_score_hist_exact,
    _fit_nampy_model_and_trace,
    _run_mgcv_trace,
)


@pytest.mark.parametrize(
    ("data_factory", "formula", "select", "score_atol", "sp_atol"),
    LINKED_ID_TRACE_CASES,
)
def test_gaussian_linked_id_reml_score_hist_matches_mgcv_supported_bases(
    data_factory,
    formula,
    select,
    score_atol,
    sp_atol,
):
    """Verify that gaussian linked id REML score hist matches mgcv supported bases."""
    data = data_factory()
    model, _ = _fit_nampy_model_and_trace(
        data,
        formula,
        "gaussian",
        "REML",
        select=select,
    )
    expected = _run_mgcv_trace(
        data,
        formula,
        "gaussian",
        "REML",
        select=select,
    )

    _assert_mgcv_score_hist_exact(model, expected, atol=score_atol)
    np.testing.assert_allclose(
        np.log(np.asarray(model.smoothing_params, dtype=np.float64)),
        np.log(np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)),
        atol=sp_atol,
        rtol=0.0,
    )
