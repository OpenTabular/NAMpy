from __future__ import annotations

# ruff: noqa: E402, I001

import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh as scipy_eigh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nampy.gam.fit.solvers.general_family.fixed_smoothing import (
    GeneralPenaltySetup,
    build_general_family_setup_state,
    sl_initial_repara,
)
from tests.families.test_general_family_mgcv_parity import (
    GAULSS_FORMULA,
    _gammals_by_data,
    _gammals_data,
    _gaulss_by_data,
    _gaulss_data,
)
from tests.mgcv_parity_utils import _fit_nampy_model_fixed_sp
from tests.optimization.test_mgcv_general_family_preoptimization_parity import (
    _run_mgcv_general_preoptimization,
)


CASES = {
    "gaulss_cr": ("gaulss", GAULSS_FORMULA, _gaulss_data, False),
    "gaulss_select_true_cr": ("gaulss", GAULSS_FORMULA, _gaulss_data, True),
    "gaulss_numeric_by": (
        "gaulss",
        ['y ~ s(x, by=z, bs="cr", k=6)', "~ 1"],
        _gaulss_by_data,
        False,
    ),
    "gammals_cr": (
        "gammals",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _gammals_data,
        False,
    ),
    "gammals_numeric_by": (
        "gammals",
        ['y ~ s(x, by=z, bs="cr", k=6)', "~ 1"],
        _gammals_by_data,
        False,
    ),
}


def _corr_matrix(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    out = np.zeros((actual.shape[1], expected.shape[1]), dtype=np.float64)
    for i in range(actual.shape[1]):
        for j in range(expected.shape[1]):
            den = float(np.linalg.norm(actual[:, i]) * np.linalg.norm(expected[:, j]))
            out[i, j] = 0.0 if den == 0.0 else float(actual[:, i] @ expected[:, j] / den)
    return out


def _variant_initial(setup, *, driver: str, lower: bool) -> np.ndarray:
    blocks = []
    for block in setup.Sl:
        if len(block.S) != 1 or block.D is None or np.asarray(block.D).ndim != 2:
            blocks.append(block)
            continue
        S = np.asarray(block.S[0], dtype=np.float64)
        if driver == "numpy":
            vals, vecs = np.linalg.eigh(S if lower else np.asarray(S.T, dtype=np.float64))
        else:
            vals, vecs = scipy_eigh(S, lower=lower, check_finite=False, driver=driver)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        rank = int(block.rank)
        ind = np.zeros(vals.size, dtype=bool)
        ind[:rank] = True
        dvals = vals.copy()
        dvals[ind] = 1.0 / np.sqrt(dvals[ind])
        dvals[~ind] = 1.0
        b = type(block)(
            start=block.start,
            stop=block.stop,
            S=block.S,
            lambda_=block.lambda_,
            repara=block.repara,
            linear=block.linear,
            rank=rank,
            ldet=block.ldet,
            ind=ind,
            D=np.asarray(vecs * dvals[np.newaxis, :], dtype=np.float64),
            Di=np.asarray(vecs.T / dvals[:, np.newaxis], dtype=np.float64),
            penalty_indices=block.penalty_indices,
        )
        blocks.append(b)
    sl = GeneralPenaltySetup(
        blocks=blocks,
        E=setup.Sl.E,
        S=setup.Sl.S,
        lambda_=setup.Sl.lambda_,
        cholesky=setup.Sl.cholesky,
    )
    return sl_initial_repara(sl, setup.X_full, both_sides=False)


def main() -> None:
    case_id = sys.argv[1] if len(sys.argv) > 1 else "gaulss_cr"
    family, formula, data_factory, select = CASES[case_id]
    data = data_factory()
    expected = _run_mgcv_general_preoptimization(
        data, formula, family, "ML", select=select
    )
    sp = np.asarray(expected["smoothing_params"], dtype=np.float64)
    gam = _fit_nampy_model_fixed_sp(data, formula, family, sp, select=select)
    actual = build_general_family_setup_state(gam, sp, score_type="ML")
    X = np.asarray(actual.X_full, dtype=np.float64)
    X_ref = np.asarray(expected["X_full"], dtype=np.float64)
    corr = _corr_matrix(X, X_ref)
    Xi = np.asarray(actual.X_initial, dtype=np.float64)
    Xi_ref = np.asarray(expected["X_initial"], dtype=np.float64)
    variants = []
    for driver in ("evr", "evd", "ev", "numpy"):
        for lower in (True, False):
            Xv = _variant_initial(actual, driver=driver, lower=lower)
            variants.append(
                {
                    "driver": driver,
                    "lower": lower,
                    "max_abs": float(np.max(np.abs(Xv - Xi_ref))),
                    "signed_max_abs": float(
                        np.max(
                            np.abs(
                                Xv
                                * np.sign(
                                    np.diag(_corr_matrix(Xv, Xi_ref))
                                )[np.newaxis, :]
                                - Xi_ref
                            )
                        )
                    ),
                }
            )
    best = [
        {
            "actual": i,
            "expected": int(np.argmax(np.abs(corr[i]))),
            "corr": float(corr[i, int(np.argmax(np.abs(corr[i])))]),
        }
        for i in range(corr.shape[0])
    ]
    initial_corr_diag = np.diag(_corr_matrix(Xi, Xi_ref)).tolist()
    sl_ref = expected["Sl"]["blocks"][0]
    d_ref = np.asarray(sl_ref["D"], dtype=np.float64)
    d_act = np.asarray(actual.Sl.blocks[0].D, dtype=np.float64)
    evals = np.linalg.eigvalsh(np.asarray(actual.Sl.blocks[0].S[0], dtype=np.float64))
    print(
        json.dumps(
            {
                "case": case_id,
                "shape": list(X.shape),
                "max_abs": float(np.max(np.abs(X - X_ref))),
                "initial_max_abs": float(np.max(np.abs(Xi - Xi_ref))),
                "initial_corr_diag": initial_corr_diag,
                "best": best,
                "variants": variants,
                "D_actual": d_act.tolist(),
                "D_expected": d_ref.tolist(),
                "eigvalsh": evals.tolist(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
