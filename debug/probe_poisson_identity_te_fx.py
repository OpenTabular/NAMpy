from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nampy.gam._model_state import (  # noqa: E402
    _design_matrix,
    _fit_intercept,
    _n_coef,
    _penalty_blocks_seq,
)
from nampy.gam.fit.linalg.stacked_qr import (  # noqa: E402
    balanced_penalty_template_sqrt_for_rank,
)
from nampy.gam.fit.penalized_system import (  # noqa: E402
    build_full_design,
    build_full_penalty_from_blocks,
)
from nampy.gam.fit.solvers.irls_core import _mgcv_null_coef, irls_core  # noqa: E402
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

    fi = _fit_intercept(gam)
    Z = np.asarray(_design_matrix(gam), dtype=np.float64)
    penalty_blocks = tuple(_penalty_blocks_seq(gam))
    X = build_full_design(Z, fit_intercept=fi)
    S = build_full_penalty_from_blocks(
        penalty_blocks=penalty_blocks,
        smoothing_params=np.asarray(gam.smoothing_params, dtype=np.float64),
        fit_intercept=fi,
        n_coef=_n_coef(gam),
    )
    rank_rows = balanced_penalty_template_sqrt_for_rank(
        penalty_blocks, fit_intercept=fi, n_coef=int(_n_coef(gam))
    )
    y = np.asarray(data["y"], dtype=np.float64)
    for fisher_only in (False, True):
        sol = irls_core(
            X,
            y,
            gam.family,
            S,
            offset=gam.offset_train_,
            weights=np.ones_like(y),
            fit_intercept=fi,
            max_iter=200,
            tol=1e-11,
            null_coef=_mgcv_null_coef(X, y, gam.family),
            fisher_scoring_only=fisher_only,
            penalty_rank_rows=rank_rows,
        )
        print(
            "direct",
            {"fisher_only": fisher_only},
            {
                "loglik": sol["loglik"],
                "deviance": sol["deviance"],
                "edf": sol["edf"],
                "converged": sol["converged"],
                "failed_step": sol["failed_step"],
                "failure_reason": sol["failure_reason"],
                "iter": sol["iter"],
                "last_trace": sol["inner_trace"][-3:],
            },
        )


if __name__ == "__main__":
    main()
