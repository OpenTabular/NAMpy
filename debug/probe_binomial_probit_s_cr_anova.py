from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nampy.gam._model_state import (  # noqa: E402
    _coef,
    _coef_column_offset,
    _fit_state,
    _summary_R,
    _term_blocks_seq,
)
from nampy.gam.fit import select_covariance_matrix  # noqa: E402
from nampy.gam.inference.anova import _smooth_test_stat, _term_edf1  # noqa: E402
from tests.gam_cartesian_matrix import MatrixCase, fit_model, make_data  # noqa: E402
from tests.mgcv_parity_utils import (  # noqa: E402
    _run_mgcv_gam_setup_assembly,
    _run_mgcv_snapshot,
)


FORMULA = 'y ~ s(x0, bs="cr", k=8, sp=1.1)'
FAMILY = {"name": "binomial", "link": "probit"}


def main() -> None:
    data = make_data("binary")
    case = MatrixCase(
        case_id="debug_binomial_probit_s_cr",
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
    setup = _run_mgcv_gam_setup_assembly(
        data,
        case.formula,
        case.family,
        case.method,
        allow_live_run=True,
    )
    print(
        "anova",
        actual["parity"]["diagnostics"]["anova_smooth"]["values"],
        expected["parity"]["diagnostics"]["anova_smooth"]["values"],
    )
    tb = [tb for tb in _term_blocks_seq(gam) if str(tb.term_type) != "parametric"][0]
    offset = _coef_column_offset(gam)
    sl = tb.coef_slice
    x_sl = slice(sl.start + offset, sl.stop + offset)
    beta = np.asarray(_coef(gam), dtype=np.float64).ravel()[sl]
    beta_expected = np.asarray(expected["fit"]["coef_full"], dtype=np.float64)[x_sl]
    cov = np.asarray(select_covariance_matrix(gam, cov="bayes"), dtype=np.float64)[
        x_sl, x_sl
    ]
    cov_expected = np.asarray(expected["fit"]["cov_bayes"], dtype=np.float64)[
        x_sl, x_sl
    ]
    rank = min(float(x_sl.stop - x_sl.start), max(_term_edf1(gam, tb), 1.0))
    fit_X = np.asarray(_fit_state(gam).X, dtype=np.float64)[:, x_sl]
    expected_X = np.asarray(expected["predictions"]["lpmatrix"], dtype=np.float64)[
        :, x_sl
    ]
    setup_X = np.asarray(setup["X"], dtype=np.float64)[:, x_sl]
    summary_R = np.asarray(_summary_R(gam), dtype=np.float64)[:, x_sl]
    print(
        "diffs",
        {
            "fit_vs_expected_X": float(np.max(np.abs(fit_X - expected_X))),
            "fit_vs_setup": float(np.max(np.abs(fit_X - setup_X))),
            "coef": float(np.max(np.abs(beta - beta_expected))),
            "cov": float(np.max(np.abs(cov - cov_expected))),
        },
    )
    for name, X, b, V in [
        ("fit_X", fit_X, beta, cov),
        ("expected_X", expected_X, beta_expected, cov_expected),
        ("setup_X", setup_X, beta_expected, cov_expected),
        ("summary_R", summary_R, beta, cov),
    ]:
        print(name, _smooth_test_stat(b, X, V, rank=rank, residual_df=-1.0))


if __name__ == "__main__":
    main()
