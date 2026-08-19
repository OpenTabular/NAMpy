from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.gam_cartesian_matrix import MatrixCase, fit_model, make_data  # noqa: E402

from nampy.gam.fit.penalized_system import (  # noqa: E402
    build_full_design,
    build_full_penalty_from_blocks,
)
from nampy.gam.fit.solvers.irls_core import _mgcv_null_coef, irls_core  # noqa: E402
from nampy.gam.linalg import (  # noqa: E402
    balanced_penalty_template_sqrt_for_rank,
)
from nampy.gam.model_state import (  # noqa: E402
    _design_matrix,
    _fit_intercept,
    _fit_result,
    _n_coef,
    _penalty_blocks_seq,
)
from tests.mgcv_parity_utils import (  # noqa: E402
    _run_mgcv_gam_setup_assembly,
    _run_mgcv_snapshot,
)

FORMULA = 'y ~ te(x0, x1, bs=["cr","cr"], k=[6,6], fx=TRUE)'
FAMILY = "binomial"


def _loglik_from_mu(family, y, mu, weights):
    return float(
        np.sum(
            weights
            * np.asarray(family.loglik_obs(y, mu, scale=1.0), dtype=np.float64)
        )
    )


def _projector(X):
    q, _ = np.linalg.qr(np.asarray(X, dtype=np.float64), mode="reduced")
    return q @ q.T


def main() -> None:
    data = make_data("binary")
    case = MatrixCase(
        case_id="debug_binomial_logit_te_fx",
        formula=FORMULA,
        family=FAMILY,
        method="fixed",
        data_kind="binary",
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
    print(
        "snapshot loglik/deviance",
        actual["fit"].get("loglik"),
        expected["fit"].get("loglik"),
        actual["fit"].get("deviance"),
        expected["fit"].get("deviance"),
    )
    print(
        "prediction max abs",
        float(
            np.max(
                np.abs(
                    np.asarray(actual["predictions"]["response"], dtype=np.float64)
                    - np.asarray(expected["predictions"]["response"], dtype=np.float64)
                )
            )
        ),
    )

    setup = _run_mgcv_gam_setup_assembly(
        data,
        case.formula,
        case.family,
        case.method,
        allow_live_run=True,
    )
    actual_lpmatrix = np.asarray(gam.predict(data, type="lpmatrix"), dtype=np.float64)
    expected_lpmatrix = np.asarray(expected["predictions"]["lpmatrix"], dtype=np.float64)
    setup_X = np.asarray(setup["X"], dtype=np.float64)
    print(
        "design shapes",
        actual_lpmatrix.shape,
        expected_lpmatrix.shape,
        setup_X.shape,
    )
    print(
        "lpmatrix max abs",
        float(np.max(np.abs(actual_lpmatrix - expected_lpmatrix))),
    )
    print(
        "setup projector max abs",
        float(np.max(np.abs(_projector(actual_lpmatrix) - _projector(setup_X)))),
    )

    y = np.asarray(data["y"], dtype=np.float64)
    weights = np.ones_like(y)
    print(
        "loglik from fitted mu",
        _loglik_from_mu(gam.family, y, _fit_result(gam).mu, weights),
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
    sol = irls_core(
        X,
        y,
        gam.family,
        S,
        offset=gam.offset_train_,
        weights=weights,
        fit_intercept=fi,
        max_iter=200,
        tol=1e-7,
        null_coef=_mgcv_null_coef(X, y, gam.family),
        penalty_rank_rows=rank_rows,
    )
    print(
        "direct",
        {
            "loglik": sol["loglik"],
            "loglik_from_report_mu": _loglik_from_mu(
                gam.family, y, np.asarray(sol["mu"], dtype=np.float64), weights
            ),
            "deviance": sol["deviance"],
            "deviance_from_report_mu": float(
                gam.family.deviance(y, np.asarray(sol["mu"], dtype=np.float64))
            ),
            "iter": sol["iter"],
            "converged": sol["converged"],
            "failed_step": sol["failed_step"],
            "failure_reason": sol["failure_reason"],
            "warnings": sol["warnings"],
            "last_trace": sol["inner_trace"][-3:],
        },
    )


if __name__ == "__main__":
    main()
