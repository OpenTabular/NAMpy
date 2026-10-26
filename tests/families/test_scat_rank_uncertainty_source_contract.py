"""Rank-deficient upstream uncertainty evidence without platform assumptions."""

import json
from pathlib import Path

import numpy as np


def test_scat_source_rank_deficient_vb_correction_capture():
    root = Path(__file__).parents[1] / "reference_fixtures/mgcv_scat_solver"
    reference = json.loads((root / "rank_vb_correction.json").read_text())
    assert reference["mgcv_version"] == "1.9.4"
    capture = reference["capture"]
    H = np.asarray(capture["H"], dtype=float)
    X = np.asarray(reference["modelX"], dtype=float)
    assert np.linalg.matrix_rank(H) == H.shape[0] - 1
    assert np.linalg.matrix_rank(X) == X.shape[1] - 1
    # The duplicate z and I(2*z) parametric columns create an exact null
    # direction. Do not assert a local Cholesky outcome: it depends on BLAS
    # rounding at the zero pivot, even when the matrices agree entrywise.
    null = np.zeros(H.shape[0])
    null[1:3] = [-2.0, 1.0]
    np.testing.assert_allclose(X @ null, 0, atol=1e-14)
    np.testing.assert_allclose(H @ null, 0, atol=1e-11)
    # This is the recorded upstream branch, not a demand that the current
    # host reproduce that floating-point branch.
    assert capture["chol_ok"] is False
    assert capture["R"] == {}  # jsonlite default encoding of R NULL.
    assert capture["correction"] == 0
