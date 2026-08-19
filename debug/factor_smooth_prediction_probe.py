"""Localize factor-smooth term-contribution parity against mgcv."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nampy.gam import GAM
from nampy.gam.model_state import _coef, _coef_full, _term_blocks_seq
from nampy.gam.predict.linear_predictor_matrix import _build_prediction_matrices
from nampy.gam.predict.predictions import (
    _prediction_term_groups,
    _term_contribution,
    _term_contribution_shift,
)
from tests.mgcv_parity_utils import _run_mgcv_snapshot


def main() -> None:
    rng = np.random.default_rng(732)
    n = 72
    x = rng.uniform(-1.3, 1.3, size=n)
    f = pd.Categorical(rng.choice(np.array(["a", "b", "c"], dtype=object), size=n))
    y = np.sin(x) + np.array(
        [{"a": 0.2, "b": -0.3, "c": 0.45}[str(value)] for value in f]
    )
    y = y + rng.normal(scale=0.05, size=n)
    data = pd.DataFrame({"y": y, "x": x, "f": f})
    formula = 'y ~ s(x, f, bs="fs", k=5, xt="cr", sp=[0.8, 0.8, 0.8])'

    gam = GAM(family="gaussian", formula=formula)
    gam.fit(data=data)
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
    Z_new, Xp = _build_prediction_matrices(gam, X_new=gam.X_)

    print("smoothing params", np.asarray(gam.smoothing_params).tolist())
    print("coef full", np.asarray(_coef_full(gam)).tolist())
    print("term groups", _prediction_term_groups(gam))
    for index, term in enumerate(_term_blocks_seq(gam)):
        raw = Z_new[:, term.coef_slice] @ np.asarray(_coef(gam))[term.coef_slice]
        shifted = _term_contribution(gam, Z_new, term)
        print(
            "term",
            index,
            "label",
            term.label,
            "type",
            term.term_type,
            "slice",
            term.coef_slice,
            "shift",
            _term_contribution_shift(gam, term),
            "raw head",
            raw[:5].tolist(),
            "shifted head",
            shifted[:5].tolist(),
        )
    print("Xp shape", Xp.shape)
    print("expected coef", expected["fit"]["coef_full"])
    print("expected terms head", expected["predictions"]["terms"][:5])
    print("expected lpmatrix shape", np.asarray(expected["predictions"]["lpmatrix"]).shape)


if __name__ == "__main__":
    main()
