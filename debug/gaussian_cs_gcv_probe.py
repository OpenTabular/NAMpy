"""Compare Gaussian ``cs`` GCV inputs at mgcv's smoothing endpoint."""

from __future__ import annotations

import json

import numpy as np
from scipy.linalg import eigh

from nampy.gam.fit.backends import solve_gaussian_given_smoothing
from nampy.gam.penalties.algebra import penalty_rescale_factor
from nampy.gam.smoothing_selection.criteria.dispatch import criterion_value
from nampy.splines.univariate.cr import add_full_rank_shrinkage
from tests.mgcv_parity_utils import (
    _fit_nampy_model_fixed_sp,
    _make_gaussian_data,
    _run_mgcv_raw_constructor,
    _run_mgcv_smoothcon_penalties,
    _run_mgcv_snapshot,
)
from tests.smooths.test_mgcv_raw_constructor_parity import _build_runtime_term


def _serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def main() -> None:
    data = _make_gaussian_data(seed=2024, n=220)
    formula = 'y ~ s(x0, bs="cs", k=8)'
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "GCV.Cp")
    sp = np.atleast_1d(
        np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    ).ravel()
    model = _fit_nampy_model_fixed_sp(data, formula, "gaussian", sp)
    sol = solve_gaussian_given_smoothing(model, model.y_, sp)
    expected_penalties = _run_mgcv_smoothcon_penalties(
        data,
        's(x0, bs="cs", k=8)',
        absorb_cons=True,
        scale_penalty=True,
    )
    actual_penalty = np.asarray(
        model.compiled_model_.compiled_penalties[0].matrix,
        dtype=np.float64,
    )
    expected_penalty = np.asarray(expected_penalties["S"][0], dtype=np.float64)
    term, _X, _feature_names = _build_runtime_term(data, formula)
    raw_cr = np.asarray(term._spline.raw_penalty_unscaled, dtype=np.float64)
    expected_raw_cr = np.asarray(
        _run_mgcv_raw_constructor(data, 's(x0, bs="cr", k=8)')["S"][0],
        dtype=np.float64,
    )
    expected_raw_cs = np.asarray(
        _run_mgcv_raw_constructor(data, 's(x0, bs="cs", k=8)')["S"][0],
        dtype=np.float64,
    )
    eig_variants = {}
    for driver in ("ev", "evd", "evr", "evx"):
        values, vectors = eigh(
            0.5 * (raw_cr + raw_cr.T),
            lower=True,
            driver=driver,
            check_finite=False,
        )
        values = values[::-1].copy()
        vectors = vectors[:, ::-1].copy()
        values[-2] = values[-3] * 0.1
        values[-1] = values[-2] * 0.1
        rebuilt = vectors @ (values[:, None] * vectors.T)
        eig_variants[driver] = float(np.max(np.abs(rebuilt - expected_raw_cs)))
    n = float(model.n_samples_)
    actual = {
        "criterion": criterion_value(model, model.y_, np.log(sp), method="gcv"),
        "rss": sol["rss"],
        "deviance": sol["deviance"],
        "trace_H": sol["trace_H"],
        "gcv_from_deviance": n
        * float(sol["deviance"])
        / (n - float(model.score_gamma) * float(sol["trace_H"])) ** 2,
        "coef_full": np.asarray(sol["coef_full"], dtype=np.float64),
        "coef_max_abs": float(
            np.max(
                np.abs(
                    np.asarray(sol["coef_full"], dtype=np.float64)
                    - np.asarray(expected["fit"]["coef_full"], dtype=np.float64)
                )
            )
        ),
        "compiled_penalty_max_abs": float(
            np.max(np.abs(actual_penalty - expected_penalty))
        ),
        "raw_cr_max_abs": float(np.max(np.abs(raw_cr - expected_raw_cr))),
        "raw_cs_max_abs": float(
            np.max(
                np.abs(
                    add_full_rank_shrinkage(raw_cr, shrink=0.1)
                    - expected_raw_cs
                )
            )
        ),
        "raw_cs_eig_variants": eig_variants,
        "nampy_penalty_scale": penalty_rescale_factor(
            term._spline.raw_basis,
            add_full_rank_shrinkage(raw_cr, shrink=0.1),
        ),
    }
    print(
        json.dumps(
            _serializable(
                {
                    "mgcv": {
                        key: expected["fit"][key]
                        for key in (
                            "criterion_value",
                            "rss",
                            "deviance",
                            "trace_H",
                            "coef_full",
                        )
                    },
                    "nampy": actual,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
