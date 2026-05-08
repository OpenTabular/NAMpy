from __future__ import annotations

# ruff: noqa: E402, I001

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nampy.gam._model_state import _design_matrix, _n_coef, _penalty_blocks_seq  # noqa: E402
from nampy.gam.families import BinomialLogitFamily  # noqa: E402
from nampy.gam.fit.linalg.stacked_qr import (  # noqa: E402
    balanced_penalty_template_sqrt_for_rank,
)
from nampy.gam.fit.penalized_system import (  # noqa: E402
    build_full_design,
    build_full_penalty_from_blocks,
)
from nampy.gam.fit.solvers.irls_core import irls_core  # noqa: E402
from nampy.gam.model.api import GAM  # noqa: E402


def main() -> None:
    x = np.linspace(-2.0, 2.0, 80, dtype=np.float64)
    y = (x > 0.0).astype(np.float64)

    gam = GAM(k=8, optimize_smoothing=False, smoothing_method="fixed")
    gam.fit(X=x[:, None], y=np.sin(x))

    X = build_full_design(_design_matrix(gam), fit_intercept=gam.fit_intercept)
    S = build_full_penalty_from_blocks(
        penalty_blocks=_penalty_blocks_seq(gam),
        smoothing_params=gam.smoothing_params,
        fit_intercept=gam.fit_intercept,
        n_coef=_n_coef(gam),
    )
    rank_rows = balanced_penalty_template_sqrt_for_rank(
        _penalty_blocks_seq(gam),
        fit_intercept=gam.fit_intercept,
        n_coef=int(_n_coef(gam)),
    )

    irls_core_module = importlib.import_module("nampy.gam.fit.solvers.irls_core")
    with patch.object(irls_core_module.np.linalg, "cond", lambda _A: 1e13):
        sol = irls_core(
            X,
            y=y,
            family=BinomialLogitFamily(),
            S=S,
            max_iter=50,
            offset=None,
            fit_intercept=gam.fit_intercept,
            penalty_rank_rows=rank_rows,
        )

    print(
        json.dumps(
            {
                "failed_step": bool(sol["failed_step"]),
                "failure_reason": sol["failure_reason"],
                "converged": bool(sol["converged"]),
                "iter": int(sol["iter"]),
                "warnings": sol["warnings"],
                "last_trace": sol["inner_trace"][-5:],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
