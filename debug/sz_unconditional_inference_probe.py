"""Inspect SZ gauge transforms used by covariance and ``summary.gam`` parity."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nampy.gam.inference.anova import _smooth_test_stat, _term_edf1
from nampy.gam.model_state import (
    _coef_column_offset,
    _coef_full,
    _cov_bayes,
    _fit_state,
    _summary_R,
    _term_blocks_seq,
)
from tests.parity.test_mgcv_prediction_inference_diagnostics_parity import (
    _case_bundle,
)


def _max_abs(value) -> float:
    arr = np.asarray(value, dtype=np.float64)
    return float(np.max(np.abs(arr))) if arr.size else 0.0


def main() -> None:
    _data, expected, model = _case_bundle("factor_smooth_sz")
    tb = next(
        term
        for term in _term_blocks_seq(model)
        if str(getattr(term, "term_type", "")) == "factor_smooth_sz"
    )
    offset = _coef_column_offset(model)
    ind = np.arange(
        offset + int(tb.coef_slice.start),
        offset + int(tb.coef_slice.stop),
        dtype=np.int64,
    )

    beta_actual = np.asarray(_coef_full(model), dtype=np.float64)[ind]
    beta_expected = np.asarray(expected["fit"]["coef_full"], dtype=np.float64)[ind]
    signs = np.where(beta_actual * beta_expected < 0.0, -1.0, 1.0)
    D = np.diag(signs)

    V_actual = np.asarray(_cov_bayes(model), dtype=np.float64)[np.ix_(ind, ind)]
    V_expected = np.asarray(
        expected["parity"]["diagnostics"]["smooth_cov_bayes"]["blocks"][0],
        dtype=np.float64,
    )
    R_actual = np.asarray(_summary_R(model), dtype=np.float64)[:, ind]
    R_expected = np.asarray(
        expected["parity"]["diagnostics"]["smooth_test_inputs"]["r_blocks"][0],
        dtype=np.float64,
    )

    expected_table = np.asarray(
        expected["parity"]["diagnostics"]["anova_smooth"]["values"],
        dtype=np.float64,
    )
    residual_df = float(
        expected["parity"]["diagnostics"]["smooth_test_inputs"]["residual_df"]
    )
    rank = min(float(R_actual.shape[1]), max(_term_edf1(model, tb), 1.0))
    stat_actual = _smooth_test_stat(
        beta_actual, R_actual, V_actual, rank, residual_df
    )[0]
    stat_expected_inputs = _smooth_test_stat(
        beta_expected, R_expected, V_expected, rank, residual_df
    )[0]

    X = np.asarray(_fit_state(model).X, dtype=np.float64)[:, ind]
    stat_design = _smooth_test_stat(beta_actual, X, V_actual, rank, residual_df)[0]

    runtime = getattr(getattr(tb, "predict_fn", None), "__self__", None)
    base = getattr(runtime, "_base_term", None)
    B0 = np.asarray(getattr(base, "basis_train", np.empty((0, 0))), dtype=np.float64)
    UZ = np.asarray(
        getattr(getattr(base, "_setup", None), "UZ", np.empty((0, 0))),
        dtype=np.float64,
    )
    max_rows = (
        np.argmax(np.abs(B0), axis=0) if B0.ndim == 2 and B0.shape[0] else np.array([])
    )
    max_signs = (
        np.sign(B0[max_rows, np.arange(B0.shape[1])])
        if B0.ndim == 2 and B0.shape[1]
        else np.array([])
    )

    print("coefficient_signs", signs.astype(int).tolist())
    print("coef_sign_aligned_max_abs", _max_abs(D @ beta_actual - beta_expected))
    print("Vp_sign_aligned_max_abs", _max_abs(D @ V_actual @ D - V_expected))
    print("R_sign_aligned_max_abs", _max_abs(R_actual @ D - R_expected))
    print("stat_actual", stat_actual)
    print("stat_expected_inputs", stat_expected_inputs)
    print("stat_design", stat_design)
    print("stat_expected_table", float(expected_table[0, 2]))
    print("base_first_row", B0[0].tolist() if B0.shape[0] else [])
    print("base_column_sums", np.sum(B0, axis=0).tolist())
    print("base_max_abs_signs", max_signs.astype(int).tolist())
    print("UZ_first_row", UZ[0].tolist() if UZ.shape[0] else [])
    print(
        "UZ_max_abs_signs",
        (
            np.sign(UZ[np.argmax(np.abs(UZ), axis=0), np.arange(UZ.shape[1])])
            .astype(int)
            .tolist()
            if UZ.ndim == 2 and UZ.shape[0]
            else []
        ),
    )

if __name__ == "__main__":
    main()
