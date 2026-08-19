from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.gam_cartesian_matrix import MatrixCase, fit_model, make_data  # noqa: E402

from tests.mgcv_parity_utils import _run_mgcv_snapshot  # noqa: E402

FORMULA = 'y ~ te(x0, x1, bs=["cr","cr"], k=[6,6], fx=TRUE)'
FAMILY = {"name": "poisson", "link": "identity"}


def main() -> None:
    data = make_data("count")
    case = MatrixCase(
        case_id="debug_poisson_identity_te_fx",
        formula=FORMULA,
        family=FAMILY,
        method="fixed",
        data_kind="count",
    )
    gam = fit_model(case, data)
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(
        data,
        case.formula,
        case.family,
        case.method,
        allow_live_run=True,
    )
    print("snapshot loglik/deviance", actual["fit"].get("loglik"), expected["fit"].get("loglik"))
    print("snapshot deviance", actual["fit"].get("deviance"), expected["fit"].get("deviance"))
    print("snapshot edf", actual["fit"].get("edf_total"), expected["fit"].get("edf_total"))
    print(
        "coef/fitted max abs",
        float(
            np.max(
                np.abs(
                    np.asarray(actual["fit"]["coef_full"], dtype=np.float64)
                    - np.asarray(expected["fit"]["coef_full"], dtype=np.float64)
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    np.asarray(actual["predictions"]["response"], dtype=np.float64)
                    - np.asarray(expected["predictions"]["response"], dtype=np.float64)
                )
            )
        ),
    )

if __name__ == "__main__":
    main()
