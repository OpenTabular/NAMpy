from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import qr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.gam_cartesian_matrix import MatrixCase, fit_model, make_data  # noqa: E402

from nampy.gam.fit import select_covariance_matrix  # noqa: E402
from nampy.gam.inference.anova import _smooth_test_stat, _term_edf1  # noqa: E402
from nampy.gam.inference.chi_square_mixtures import psum_chisq  # noqa: E402
from nampy.gam.linalg import symmetric_eigh, symmetrize_matrix  # noqa: E402
from nampy.gam.model_state import (  # noqa: E402
    _coef,
    _coef_column_offset,
    _fit_state,
    _summary_R,
    _term_blocks_seq,
)
from tests.mgcv_parity_utils import (  # noqa: E402
    _run_mgcv_gam_setup_assembly,
    _run_mgcv_snapshot,
)

FORMULA = 'y ~ s(f, x0, bs="sz", k=6, sp=[1.0,1.2,1.4,1.6])'
FAMILY = "binomial"


def _test_stat_scipy_pivot(p, X, V, rank):
    _q, R, piv = qr(np.asarray(X, dtype=np.float64), mode="economic", pivoting=True)
    V = np.asarray(V, dtype=np.float64)
    Vt = R @ V[np.ix_(piv, piv)] @ R.T
    evals, evecs = symmetric_eigh(symmetrize_matrix(Vt), descending=True)
    signs = np.sign(evecs[0, :])
    signs[signs == 0.0] = 1.0
    evecs = evecs * signs
    k = max(0, int(np.floor(rank)))
    nu = abs(float(rank) - k)
    k1 = k + 1 if nu > 0.0 else k
    r_est = int(
        np.sum(evals > max(float(np.max(evals)), 0.0) * np.finfo(np.float64).eps**0.9)
    )
    if r_est < k1:
        k1 = k = r_est
        nu = 0.0
        rank = float(r_est)
    vec = evecs[:, :k1].copy()
    if nu > 0.0 and k > 0:
        if k > 1:
            vec[:, : k - 1] = vec[:, : k - 1] / np.sqrt(evals[: k - 1])
        b12 = np.sqrt(max(0.5 * nu * (1.0 - nu), 0.0))
        B = np.array([[1.0, b12], [b12, nu]], dtype=np.float64)
        ev = np.diag(evals[k - 1 : k1] ** -0.5)
        B = ev @ B @ ev
        eb_vals, eb_vecs = np.linalg.eigh(B)
        rB = eb_vecs @ np.diag(np.sqrt(eb_vals)) @ eb_vecs.T
        vec1 = vec.copy()
        vec1[:, k - 1 : k1] = (
            rB @ np.diag([-1.0, 1.0]) @ vec[:, k - 1 : k1].T
        ).T
        vec[:, k - 1 : k1] = (rB @ vec[:, k - 1 : k1].T).T
    else:
        vec = vec / np.sqrt(evals[:k1])
        vec1 = vec.copy()
        if k == 1:
            rank = 1.0
    Rp = R @ np.asarray(p, dtype=np.float64).ravel()
    d = float(np.sum((vec.T @ Rp) ** 2))
    d1 = float(np.sum((vec1.T @ Rp) ** 2))
    if nu > 0.0:
        val = np.ones(k1, dtype=np.float64)
        rp = nu + 1.0
        val[k - 1] = (rp + np.sqrt(rp * (2.0 - rp))) / 2.0
        val[k1 - 1] = rp - val[k - 1]
        pval = 0.5 * (float(psum_chisq(d, val)) + float(psum_chisq(d1, val)))
    else:
        pval = 2.0
    return d, float(rank), pval


def main() -> None:
    data = make_data("binary")
    case = MatrixCase(
        case_id="debug_binomial_logit_sz_default",
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
    pred_X = np.asarray(gam.predict(data, type="lpmatrix"), dtype=np.float64)[:, x_sl]
    expected_X = np.asarray(expected["predictions"]["lpmatrix"], dtype=np.float64)[
        :, x_sl
    ]
    setup_X = np.asarray(setup["X"], dtype=np.float64)[:, x_sl]
    summary_R = np.asarray(_summary_R(gam), dtype=np.float64)[:, x_sl]
    print(
        "matrix diffs",
        {
            "fit_vs_pred": float(np.max(np.abs(fit_X - pred_X))),
            "pred_vs_mgcv": float(np.max(np.abs(pred_X - expected_X))),
            "fit_vs_setup": float(np.max(np.abs(fit_X - setup_X))),
            "setup_vs_mgcv_pred": float(np.max(np.abs(setup_X - expected_X))),
            "cov_vs_mgcv": float(np.max(np.abs(cov - cov_expected))),
            "coef_vs_mgcv": float(np.max(np.abs(beta - beta_expected))),
        },
    )
    for name, X in [
        ("fit_X", fit_X),
        ("pred_X", pred_X),
        ("expected_X", expected_X),
        ("setup_X", setup_X),
        ("summary_R", summary_R),
    ]:
        print(name, _smooth_test_stat(beta, X, cov, rank=rank, residual_df=-1.0))
    print(
        "expected_X_expected_cov",
        _smooth_test_stat(beta, expected_X, cov_expected, rank=rank, residual_df=-1.0),
    )
    print(
        "expected_all",
        _smooth_test_stat(
            beta_expected, expected_X, cov_expected, rank=rank, residual_df=-1.0
        ),
    )
    print(
        "test_stat_scipy_pivot",
        _test_stat_scipy_pivot(beta_expected, expected_X, cov_expected, rank),
    )
    for pivoting in (False, True):
        if pivoting:
            _q, R, piv = qr(expected_X, mode="economic", pivoting=True)
            R_nat = np.zeros_like(R)
            R_nat[:, piv] = R
            cov_piv = cov_expected[np.ix_(piv, piv)]
            print(
                "scipy_pivot_R_cov_piv",
                _smooth_test_stat(beta, R, cov_piv, rank=rank, residual_df=-1.0),
            )
            mats = {"scipy_pivot_R": R, "scipy_pivot_R_nat": R_nat}
        else:
            _q, R = qr(expected_X, mode="economic", pivoting=False)
            mats = {"scipy_unpivot_R": R}
        for name, Rm in mats.items():
            print(
                name,
                _smooth_test_stat(beta, Rm, cov_expected, rank=rank, residual_df=-1.0),
            )


if __name__ == "__main__":
    main()
