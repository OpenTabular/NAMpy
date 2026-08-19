"""Localize transformed ``cs`` penalty parity through the smoothCon stages."""

from __future__ import annotations

import json

import numpy as np
from scipy.linalg import eigh

from nampy.gam.penalties.algebra import penalty_rescale_factor
from nampy.gam.splines.univariate.cr import add_full_rank_shrinkage
from tests.mgcv_parity_utils import (
    _run_mgcv_raw_constructor,
    _run_mgcv_smoothcon_penalties,
)
from tests.parity.test_mgcv_output_parity import _make_gaussian_univariate_data
from tests.smooths.test_mgcv_raw_constructor_parity import _build_runtime_term


def _max_abs(left, right) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape and left.size == right.size:
        left = left.ravel()
        right = right.ravel()
    return float(np.max(np.abs(left - right)))


def main() -> None:
    data = _make_gaussian_univariate_data(seed=551, n=150)
    formula = 'y ~ s(I(x + 0.15 * x**2), bs="cs", k=8, sp=1.1)'
    term, _X, _feature_names = _build_runtime_term(data, formula)
    smooth_expr = formula.split("~", 1)[1].strip()

    expected_cr = _run_mgcv_raw_constructor(
        data,
        smooth_expr.replace('bs="cs"', 'bs="cr"'),
    )
    expected_cs = _run_mgcv_raw_constructor(data, smooth_expr)
    expected_scaled = _run_mgcv_smoothcon_penalties(
        data,
        smooth_expr,
        absorb_cons=True,
        scale_penalty=True,
    )

    raw_cr = np.asarray(term._spline.raw_penalty_unscaled, dtype=np.float64)
    expected_raw_cr = np.asarray(expected_cr["S"][0], dtype=np.float64)
    max_index = np.unravel_index(
        int(np.argmax(np.abs(raw_cr - expected_raw_cr))), raw_cr.shape
    )
    shrunk = add_full_rank_shrinkage(raw_cr, shrink=0.1)
    compiled = np.asarray(term.penalties[0], dtype=np.float64)
    expected_raw_cs = np.asarray(expected_cs["S"][0], dtype=np.float64)
    eig_variants = {}
    for source_name, source in (
        ("nampy", raw_cr),
        ("mgcv", expected_raw_cr),
    ):
        source = 0.5 * (source + source.T)
        for driver in ("ev", "evd", "evr", "evx"):
            values, vectors = eigh(
                source, lower=True, driver=driver, check_finite=False
            )
            values = values[::-1].copy()
            vectors = vectors[:, ::-1].copy()
            values[-2] = values[-3] * 0.1
            values[-1] = values[-2] * 0.1
            rebuilt = vectors @ (values[:, None] * vectors.T)
            eig_variants[f"{source_name}_{driver}"] = _max_abs(
                rebuilt, expected_raw_cs
            )
    payload = {
        "raw_cr_max_abs": _max_abs(raw_cr, expected_cr["S"][0]),
        "raw_cr_transpose_max_abs": _max_abs(raw_cr, expected_raw_cr.T),
        "raw_cr_max_index": [int(value) for value in max_index],
        "raw_cr_at_max": float(raw_cr[max_index]),
        "mgcv_raw_cr_at_max": float(expected_raw_cr[max_index]),
        "raw_F_max_abs": _max_abs(term._spline.F, expected_cr["extra"]["F"]),
        "knots_max_abs": _max_abs(term._spline.knots, expected_cr["extra"]["xp"]),
        "raw_cs_max_abs": _max_abs(shrunk, expected_cs["S"][0]),
        "raw_cs_eig_variants": eig_variants,
        "compiled_max_abs": _max_abs(compiled, expected_scaled["S"][0]),
        "nampy_scale": penalty_rescale_factor(term._spline.raw_basis, shrunk),
        "nampy_raw_eigenvalues": np.linalg.eigvalsh(0.5 * (raw_cr + raw_cr.T))[
            ::-1
        ].tolist(),
        "mgcv_raw_eigenvalues": np.linalg.eigvalsh(
            0.5 * (expected_raw_cr + expected_raw_cr.T)
        )[::-1].tolist(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
