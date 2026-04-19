from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.stats import chi2, f, ncx2

from ...mgcv_utils.davies import DaviesAlgorithm
from .._mgcv_constants import LOG_GUARD_MIN
from .._model_state import (
    _coef,
    _coef_column_offset,
    _coef_full,
    _deviance,
    _edf2,
    _edf_by_term,
    _edf_total,
    _fit_scale,
    _fit_state,
    _H_coef,
    _require_fitted,
    _summary_R,
    _term_blocks_seq,
)
from ..engine import select_covariance_matrix


def _scale_estimated(model) -> bool:
    if str(getattr(model.family, "family_class", "")).lower() == "general":
        # mgcv general.family fits carry fixed scale 1 in summary/anova output.
        return False
    return getattr(model.family, "known_scale", None) is None


def _formula_term_label(tb) -> str:
    metadata = dict(getattr(tb, "metadata", {}) or {})
    formula_term = metadata.get("formula_term", None)
    return str(getattr(tb, "label", "")) if formula_term is None else str(formula_term)


def _parametric_term_groups(model):
    groups = []
    for tb in _term_blocks_seq(model):
        if str(getattr(tb, "term_type", "")) != "parametric":
            continue
        label = _formula_term_label(tb)
        key = ("parametric", label)
        if groups and groups[-1]["key"] == key:
            groups[-1]["blocks"].append(tb)
            continue
        groups.append({"key": key, "label": label, "blocks": [tb]})
    return groups


def _term_combined_penalty_matrix(tb) -> np.ndarray | None:
    specs = tuple(getattr(tb, "penalty_specs", ()) or ())
    if len(specs) == 0:
        return None
    total = None
    for spec in specs:
        Si = np.asarray(getattr(spec, "matrix", None), dtype=np.float64)
        total = Si.copy() if total is None else total + Si
    return None if total is None else np.asarray(total, dtype=np.float64)


def _term_uses_retest(tb, summary_R) -> bool:
    if summary_R is None or str(getattr(tb, "term_type", "")) == "parametric":
        return False
    if str(getattr(tb, "term_type", "")) == "random_effect":
        return True
    total_penalty = _term_combined_penalty_matrix(tb)
    if total_penalty is None:
        return False
    width = int(total_penalty.shape[0])
    return int(np.linalg.matrix_rank(total_penalty)) >= width


def _mroot_psd(A: np.ndarray) -> np.ndarray:
    A = 0.5 * (np.asarray(A, dtype=np.float64) + np.asarray(A, dtype=np.float64).T)
    if A.size == 0:
        return np.zeros((A.shape[0], 0), dtype=np.float64)
    evals, evecs = np.linalg.eigh(A)
    # Keep the cutoff relative to the matrix scale only. mgcv::mroot() with
    # `chol(..., tol=0)` preserves tiny positive directions for exact-fit
    # cases like low-rank MRFs, and absolute `1.0 * eps` flooring drops them.
    tol = (
        max(float(np.max(evals)) if evals.size else 0.0, 0.0)
        * np.finfo(np.float64).eps
    )
    keep = evals > tol
    if not np.any(keep):
        return np.zeros((A.shape[0], 0), dtype=np.float64)
    return np.asarray(evecs[:, keep] * np.sqrt(evals[keep]), dtype=np.float64)


def _liu2(
    x: float | np.ndarray,
    lb: np.ndarray,
    *,
    df: np.ndarray | None = None,
    lower_tail: bool = False,
) -> float | np.ndarray:
    """
    Mirror mgcv/R/mgcv.r::liu2() for central chi-square mixtures.
    """
    q = np.asarray(x, dtype=np.float64)
    scalar = q.ndim == 0
    q = q.reshape(1) if scalar else q.copy()

    lb = np.asarray(lb, dtype=np.float64).ravel()
    if df is None:
        h = np.ones(lb.size, dtype=np.float64)
    else:
        h = np.asarray(df, dtype=np.float64).ravel()
        if h.size == 1:
            h = np.repeat(h, lb.size)
    if h.size != lb.size:
        raise ValueError("lambda and h should have the same length.")

    lh = lb * h
    mu_q = float(np.sum(lh))

    lh = lh * lb
    c2 = float(np.sum(lh))

    lh = lh * lb
    c3 = float(np.sum(lh))

    xpos = q > 0.0
    out = np.ones_like(q, dtype=np.float64)
    if (not np.any(xpos)) or c2 <= 0.0:
        return float(out[0]) if scalar else out

    s1 = c3 / np.power(c2, 1.5)
    s2 = float(np.sum(lh * lb)) / (c2 * c2)
    sig_q = np.sqrt(2.0 * c2)
    t = (q[xpos] - mu_q) / sig_q

    if s1 * s1 > s2:
        a = 1.0 / (s1 - np.sqrt(s1 * s1 - s2))
        delta = s1 * a * a * a - a * a
        l_df = a * a - 2.0 * delta
    else:
        if c3 == 0.0:
            return float(out[0]) if scalar else out
        a = 1.0 / s1
        delta = 0.0
        l_df = (c2 * c2 * c2) / (c3 * c3)

    mu_x = l_df + delta
    sig_x = np.sqrt(2.0) * a
    z = t * sig_x + mu_x
    if lower_tail:
        out[xpos] = ncx2.cdf(z, df=l_df, nc=delta)
    else:
        out[xpos] = ncx2.sf(z, df=l_df, nc=delta)
    return float(out[0]) if scalar else out


def _psum_chisq(
    q: float | np.ndarray,
    lb: np.ndarray,
    *,
    df: np.ndarray | None = None,
    nc: np.ndarray | None = None,
    sigz: float = 0.0,
    lower_tail: bool = False,
    tol: float = 2e-5,
    nlim: int = 100000,
) -> float | np.ndarray:
    """
    Mirror mgcv/R/mgcv.r::psum.chisq() using the existing Davies port.
    """
    x = np.asarray(q, dtype=np.float64)
    scalar = x.ndim == 0
    x = x.reshape(1) if scalar else x.copy()

    lb = np.asarray(lb, dtype=np.float64).ravel()
    r = int(lb.size)
    if r <= 0 or np.all(lb == 0.0):
        raise ValueError("at least one element of lb must be non-zero")

    if df is None:
        h = np.ones(r, dtype=np.int64)
    else:
        h = np.asarray(df, dtype=np.int64).ravel()
        if h.size == 1:
            h = np.repeat(h, r)
    if nc is None:
        delta = np.zeros(r, dtype=np.float64)
    else:
        delta = np.asarray(nc, dtype=np.float64).ravel()
        if delta.size == 1:
            delta = np.repeat(delta, r)
    if h.size != r or delta.size != r:
        raise ValueError("lengths of lb, df and nc must match")
    if np.any(h < 1):
        raise ValueError("df must be positive integers")

    solver = DaviesAlgorithm()
    out = np.empty_like(x, dtype=np.float64)
    sigz = max(float(sigz), 0.0)
    central = np.all(delta == 0.0)

    for i, qi in enumerate(x):
        cprob, _trace, ifault = solver.davies(
            lb=lb,
            nc=delta,
            n=h,
            r=r,
            sigma=sigz,
            c_val=float(qi),
            lim=int(nlim),
            acc=float(tol),
        )
        if ifault in (0, 2):
            out[i] = float(cprob if lower_tail else 1.0 - cprob)
        elif central:
            out[i] = float(_liu2(qi, lb, df=h, lower_tail=lower_tail))
        else:
            out[i] = np.nan

    return float(out[0]) if scalar else out


def _retest_like_stat(
    model,
    tb,
    *,
    residual_df: float,
    scale_estimated: bool,
    cov_freq: np.ndarray | None,
):
    fit_state = _fit_state(model)
    summary_R = _summary_R(model)
    if fit_state is None or summary_R is None or cov_freq is None:
        return None

    penalty = getattr(fit_state, "penalty_matrix", None)
    if penalty is None:
        return None

    penalty = np.asarray(penalty, dtype=np.float64)
    q = int(penalty.shape[0])
    if penalty.shape != (q, q):
        return None

    root_penalty = _mroot_psd(penalty)
    LRB = np.vstack([np.asarray(summary_R, dtype=np.float64), root_penalty.T])

    offset = _coef_column_offset(model)
    ind = np.arange(offset + tb.coef_slice.start, offset + tb.coef_slice.stop, dtype=int)
    keep = np.setdiff1d(np.arange(q, dtype=int), ind, assume_unique=True)
    perm = np.concatenate([keep, ind])
    LRB = np.asarray(LRB[:, perm], dtype=np.float64)
    Rm_full = qr(LRB, mode="economic", pivoting=False)[1]
    block = np.arange(q - ind.size, q, dtype=int)
    Rm = np.asarray(Rm_full[np.ix_(block, block)], dtype=np.float64)

    scale = max(float(_fit_scale(model)), LOG_GUARD_MIN)
    Ve_i = np.asarray(cov_freq[np.ix_(ind, ind)], dtype=np.float64)

    coef_full = np.asarray(_coef_full(model), dtype=np.float64).ravel()
    b_hat = coef_full[ind]
    stat = float(np.sum((Rm @ b_hat) ** 2) / scale)

    # mgcv::reTest() forms `crossprod(Rm %*% B) / sig2` with
    # `B <- mroot(Ve[ind, ind])`. Computing the invariant
    # `Rm %*% Ve %*% t(Rm) / sig2` avoids square-root orientation drift here.
    RBt = np.asarray((Rm @ Ve_i @ Rm.T) / scale, dtype=np.float64)
    RBt = 0.5 * (RBt + RBt.T)
    ev = np.linalg.eigvalsh(RBt)
    ev = np.clip(np.asarray(ev, dtype=np.float64), 0.0, None)
    # The direct `Ve` route needs a slightly stronger cutoff than the
    # idealized `mroot()` path to mirror mgcv's pivoted-Cholesky rank drop.
    tol = (
        max(float(np.max(ev)) if ev.size else 0.0, 0.0)
        * np.finfo(np.float64).eps ** 0.5
    )
    ev = np.asarray(ev[ev > tol], dtype=np.float64)
    rank = int(ev.size)
    if rank <= 0:
        return 0.0, 0.0, np.nan

    if scale_estimated and residual_df > 0.0:
        k = max(1, int(np.round(residual_df)))
        p_value = float(
            _psum_chisq(
                0.0,
                np.concatenate([ev, np.array([-stat / k], dtype=np.float64)]),
                df=np.concatenate(
                    [np.ones(rank, dtype=np.int64), np.array([k], dtype=np.int64)]
                ),
            )
        )
    else:
        p_value = float(_psum_chisq(stat, ev))
    return stat, float(rank), p_value


@dataclass(frozen=True)
class AnovaGAMSingle:
    family_name: str
    link_name: str
    covariance: str
    dispersion: float
    residual_df: float
    parametric_table: pd.DataFrame
    smooth_table: pd.DataFrame


@dataclass(frozen=True)
class AnovaGAMComparison:
    family_name: str
    test: str | None
    table: pd.DataFrame


def _residual_df(model) -> float:
    # Mirror mgcv/R/mgcv.r::summary.gam:
    #   residual.df <- length(object$y) - sum(object$edf)
    return float(model.n_samples_) - float(_edf_total(model))


def _edf1_vector(model) -> np.ndarray:
    H = np.asarray(_H_coef(model), dtype=np.float64)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("Model does not expose a square coefficient hat matrix.")
    return 2.0 * np.diag(H) - np.sum(H * H.T, axis=1)


def _term_edf1(model, tb) -> float:
    sl = tb.coef_slice
    offset = _coef_column_offset(model)
    edf1 = _edf1_vector(model)
    return float(np.sum(edf1[sl.start + offset : sl.stop + offset]))


def _residual_df_approx_mgcv(model) -> float:
    dfc = 0.0
    edf2 = _edf2(model)
    if edf2 is not None:
        dfc = float(np.sum(np.asarray(edf2, dtype=np.float64))) - float(
            _edf_total(model)
        )
    return float(model.n_samples_) - float(np.sum(_edf1_vector(model))) - dfc


def _stable_wald_stat(beta: np.ndarray, cov: np.ndarray) -> tuple[float, int]:
    beta = np.asarray(beta, dtype=np.float64).ravel()
    cov = np.asarray(cov, dtype=np.float64)
    if beta.size == 0:
        return 0.0, 0

    if cov.shape != (beta.size, beta.size):
        raise ValueError(
            "Coefficient covariance block shape does not match term width."
        )

    rank = int(np.linalg.matrix_rank(cov))
    if rank == 0:
        return 0.0, 0

    cov_pinv = np.linalg.pinv(cov, hermitian=True)
    stat = float(beta @ cov_pinv @ beta)
    return max(stat, 0.0), rank


def _wald_p_value(
    stat: float, ref_df: float, residual_df: float, *, scale_estimated: bool
) -> tuple[str, float]:
    if not np.isfinite(stat) or not np.isfinite(ref_df) or ref_df <= 0.0:
        return ("F" if scale_estimated else "ChiSq"), np.nan

    if scale_estimated and np.isfinite(residual_df) and residual_df > 0.0:
        f_stat = float(stat / ref_df)
        return "F", float(f.sf(f_stat, ref_df, residual_df))

    return "ChiSq", float(chi2.sf(stat, ref_df))


def _smooth_test_stat(
    p: np.ndarray, X: np.ndarray, V: np.ndarray, rank: float, residual_df: float
) -> tuple[float, float, float]:
    X = np.asarray(X, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64).ravel()
    if X.ndim != 2 or V.shape != (X.shape[1], X.shape[1]) or p.size != X.shape[1]:
        raise ValueError("Smooth test inputs have inconsistent shapes.")

    # Mirror mgcv/R/mgcv.r::testStat():
    # `Xt` is already the fitted `R` block, so the downstream `qr(Xt, tol=0)`
    # path should stay unpivoted here.
    _, R = qr(X, mode="economic", pivoting=False)
    Vt = R @ V @ R.T
    Vt = 0.5 * (Vt + Vt.T)
    evals, evecs = np.linalg.eigh(Vt)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    if evecs.size > 0:
        signs = np.sign(evecs[0, :])
        signs[signs == 0.0] = 1.0
        evecs = evecs * signs

    tol = (
        max(float(np.max(np.abs(evals))) if evals.size else 0.0, 1.0)
        * np.finfo(np.float64).eps ** 0.9
    )
    r_est = int(np.sum(evals > tol))

    k = max(0, int(np.floor(rank)))
    nu = abs(float(rank) - k)
    k1 = k + 1 if nu > 0.0 else k
    if r_est < k1:
        k1 = r_est
        k = r_est
        nu = 0.0
        rank = float(r_est)

    if k1 <= 0:
        return 0.0, 0.0, np.nan

    vec = evecs[:, :k1].copy()
    if nu > 0.0 and k > 0:
        if k > 1:
            vec[:, : (k - 1)] = vec[:, : (k - 1)] / np.sqrt(
                np.clip(evals[: (k - 1)], LOG_GUARD_MIN, None)
            )
        b12 = np.sqrt(max(0.5 * nu * (1.0 - nu), 0.0))
        B = np.array([[1.0, b12], [b12, nu]], dtype=np.float64)
        ev = np.diag(np.power(np.clip(evals[(k - 1) : k1], LOG_GUARD_MIN, None), -0.5))
        B = ev @ B @ ev
        eb_vals, eb_vecs = np.linalg.eigh(B)
        rB = eb_vecs @ np.diag(np.sqrt(np.clip(eb_vals, 0.0, None))) @ eb_vecs.T
        vec1 = vec.copy()
        vec1[:, (k - 1) : k1] = (rB @ np.diag([-1.0, 1.0]) @ vec[:, (k - 1) : k1].T).T
        vec[:, (k - 1) : k1] = (rB @ vec[:, (k - 1) : k1].T).T
    else:
        scale = np.sqrt(np.clip(evals[:k1], LOG_GUARD_MIN, None))
        vec = vec / scale
        vec1 = vec.copy()
        if k == 1:
            rank = 1.0

    Rp = R @ p
    d = float(np.sum((vec.T @ Rp) ** 2))
    d1 = float(np.sum((vec1.T @ Rp) ** 2))
    rank1 = 1.0 if nu > 0.0 and k1 == 1 else float(rank)

    if nu > 0.0:
        if k1 == 1:
            rank1 = 1.0
            val = np.array([1.0], dtype=np.float64)
        else:
            val = np.ones(k1, dtype=np.float64)
            rp = nu + 1.0
            val[k - 1] = (rp + np.sqrt(rp * (2.0 - rp))) / 2.0
            val[k1 - 1] = rp - val[k - 1]

        if residual_df > 0.0:
            k0 = max(1, int(np.round(residual_df)))
            pval = 0.5 * (
                float(
                    _psum_chisq(
                        0.0,
                        np.concatenate([val, np.array([-d / k0], dtype=np.float64)]),
                        df=np.concatenate(
                            [
                                np.ones(val.size, dtype=np.int64),
                                np.array([k0], dtype=np.int64),
                            ]
                        ),
                    )
                )
                + float(
                    _psum_chisq(
                        0.0,
                        np.concatenate([val, np.array([-d1 / k0], dtype=np.float64)]),
                        df=np.concatenate(
                            [
                                np.ones(val.size, dtype=np.int64),
                                np.array([k0], dtype=np.int64),
                            ]
                        ),
                    )
                )
            )
        else:
            pval = 0.5 * (
                float(_psum_chisq(d, val)) + float(_psum_chisq(d1, val))
            )
    else:
        pval = 2.0

    if pval > 1.0:
        if residual_df > 0.0 and rank1 > 0.0:
            pval = 0.5 * (
                float(f.sf(d / rank1, rank1, residual_df))
                + float(f.sf(d1 / rank1, rank1, residual_df))
            )
        else:
            pval = 0.5 * (
                float(chi2.sf(d, rank1)) + float(chi2.sf(d1, rank1))
            )
    return d, rank1, min(max(pval, 0.0), 1.0)


def _term_table(model, *, freq: bool, dispersion: float | None) -> AnovaGAMSingle:
    scale_est = _scale_estimated(model)
    resid_df = _residual_df(model)
    disp = float(_fit_scale(model) if dispersion is None else dispersion)

    V_para = select_covariance_matrix(model, cov=("freq" if freq else "bayes"))
    V_smooth = select_covariance_matrix(model, cov="bayes")
    try:
        V_freq = select_covariance_matrix(model, cov="freq")
    except Exception:
        V_freq = None

    beta = np.asarray(_coef(model), dtype=np.float64).ravel()
    edf_by_term = np.asarray(_edf_by_term(model), dtype=np.float64).ravel()
    summary_R = _summary_R(model)
    fit_state = _fit_state(model)

    param_rows: list[dict[str, object]] = []
    smooth_rows: list[dict[str, object]] = []

    x_offset = _coef_column_offset(model)
    for group in _parametric_term_groups(model):
        beta_cols = []
        full_cols = []
        for tb in group["blocks"]:
            beta_cols.extend(range(tb.coef_slice.start, tb.coef_slice.stop))
            full_cols.extend(range(tb.coef_slice.start + x_offset, tb.coef_slice.stop + x_offset))

        beta_i = np.asarray(beta[np.asarray(beta_cols, dtype=int)], dtype=np.float64)
        cov_i = (
            None
            if V_para is None
            else np.asarray(V_para[np.ix_(full_cols, full_cols)], dtype=np.float64)
        )
        stat, rank = (
            (np.nan, int(beta_i.size))
            if cov_i is None
            else _stable_wald_stat(beta_i, cov_i)
        )
        ref_df = float(rank)
        test_name, p_value = _wald_p_value(
            stat, ref_df, resid_df, scale_estimated=scale_est
        )
        stat_out = (
            float(stat / ref_df)
            if (scale_est and np.isfinite(stat) and ref_df > 0.0)
            else float(stat)
        )
        param_rows.append(
            {
                "label": str(group["label"]),
                "df": ref_df,
                "wald_stat": stat_out,
                "p_value": p_value,
                "test": test_name,
                "covariance": "freq" if freq else "bayes",
                "dispersion": disp,
            }
        )

    for i, tb in enumerate(_term_blocks_seq(model)):
        if str(getattr(tb, "term_type", "")) == "parametric":
            continue

        sl = tb.coef_slice
        x_sl = slice(sl.start + x_offset, sl.stop + x_offset)
        beta_i = beta[sl]
        edf_i = float(edf_by_term[i]) if i < edf_by_term.size else float(beta_i.size)

        cov_i = (
            None
            if V_smooth is None
            else np.asarray(V_smooth[x_sl, x_sl], dtype=np.float64)
        )
        if cov_i is None:
            stat, ref_df, p_value = np.nan, max(edf_i, 1.0), np.nan
        elif _term_uses_retest(tb, summary_R):
            res = _retest_like_stat(
                model,
                tb,
                residual_df=resid_df,
                scale_estimated=scale_est,
                cov_freq=V_freq,
            )
            if res is None:
                stat, ref_df, p_value = np.nan, max(edf_i, 1.0), np.nan
            else:
                stat, ref_df, p_value = res
        else:
            x_start = int(x_sl.start)
            x_stop = int(x_sl.stop)
            X_i = np.asarray(
                (
                    summary_R[:, x_start:x_stop]
                    if summary_R is not None
                    else fit_state.X[:, x_start:x_stop]
                ),
                dtype=np.float64,
            )
            edf1_i = _term_edf1(model, tb)
            stat, ref_df, p_value = _smooth_test_stat(
                beta_i,
                X_i,
                cov_i,
                rank=min(float(X_i.shape[1]), max(edf1_i, 1.0)),
                residual_df=(resid_df if scale_est else -1.0),
            )
        test_name = "F" if scale_est else "ChiSq"
        stat_out = (
            float(stat / ref_df)
            if (scale_est and np.isfinite(stat) and ref_df > 0.0)
            else float(stat)
        )
        smooth_rows.append(
            {
                "label": str(tb.label),
                "edf": edf_i,
                "ref_df": ref_df,
                "wald_stat": stat_out,
                "p_value": p_value,
                "test": test_name,
                "term_type": str(tb.term_type),
                "basis_name": str(tb.basis_name),
                "covariance": "bayes",
                "dispersion": disp,
            }
        )

    return AnovaGAMSingle(
        family_name=str(model.family.name),
        link_name=str(model.family.link_name),
        covariance=("freq" if freq else "bayes"),
        dispersion=disp,
        residual_df=resid_df,
        parametric_table=pd.DataFrame(
            param_rows,
            columns=[
                "label",
                "df",
                "wald_stat",
                "p_value",
                "test",
                "covariance",
                "dispersion",
            ],
        ),
        smooth_table=pd.DataFrame(
            smooth_rows,
            columns=[
                "label",
                "edf",
                "ref_df",
                "wald_stat",
                "p_value",
                "test",
                "term_type",
                "basis_name",
                "covariance",
                "dispersion",
            ],
        ),
    )


def _comparison_table(
    models: tuple, *, test: str | None, dispersion: float | None
) -> AnovaGAMComparison:
    family_name = str(models[0].family.name)
    method_name = str(getattr(models[0], "_optim_method", "")).lower()
    n_samples = int(models[0].n_samples_)

    for model in models:
        _require_fitted(model)
        if str(model.family.name) != family_name:
            raise ValueError(
                "anova.gam multi-model comparisons require the same family."
            )
        if int(model.n_samples_) != n_samples:
            raise ValueError(
                "anova.gam multi-model comparisons require the same sample size."
            )
        if str(getattr(model, "_optim_method", "")).lower() != method_name:
            raise ValueError(
                "anova.gam multi-model comparisons require the same smoothing selection method."
            )

    test_name = None if test is None else str(test).strip().lower()
    if test_name == "lrt":
        test_name = "chisq"
    if test_name not in {None, "chisq", "f", "cp"}:
        raise ValueError("test must be one of None, 'Chisq', 'LRT', 'F', or 'Cp'.")

    disp = float(_fit_scale(models[-1]) if dispersion is None else dispersion)
    chosen_test = test_name
    extra_cols: list[str] = []
    if chosen_test == "f":
        extra_cols = ["F", "Pr(>F)"]
    elif chosen_test == "chisq":
        extra_cols = ["Pr(>Chi)"]
    elif chosen_test == "cp":
        extra_cols = ["Cp"]

    rows: list[dict[str, object]] = []
    prev = None
    for _idx, model in enumerate(models):
        resid_df = _residual_df_approx_mgcv(model)
        row: dict[str, object] = {
            "Resid. Df": resid_df,
            "Resid. Dev": float(_deviance(model)),
            "Df": np.nan,
            "Deviance": np.nan,
        }
        for col in extra_cols:
            row[col] = np.nan

        if prev is not None:
            df_diff = float(_residual_df_approx_mgcv(prev)) - float(resid_df)
            dev_diff = float(_deviance(prev)) - float(_deviance(model))
            row["Df"] = df_diff
            row["Deviance"] = dev_diff

            if df_diff > 0.0 and dev_diff >= 0.0:
                if chosen_test == "cp":
                    row["Cp"] = float(
                        (
                            float(model.smoothing_score_)
                            if model.smoothing_score_ is not None
                            else _deviance(model)
                        )
                        - (
                            float(prev.smoothing_score_)
                            if prev.smoothing_score_ is not None
                            else _deviance(prev)
                        )
                    )
                elif chosen_test == "f":
                    denom_df = max(
                        float(resid_df),
                        1.0,
                    )
                    denom = float(_deviance(model)) / denom_df
                    stat = (
                        np.nan if denom <= 0.0 else float((dev_diff / df_diff) / denom)
                    )
                    row["F"] = stat
                    row["Pr(>F)"] = (
                        float(f.sf(stat, df_diff, denom_df))
                        if np.isfinite(stat)
                        else np.nan
                    )
                elif chosen_test == "chisq":
                    stat = float(dev_diff / disp) if disp > 0.0 else np.nan
                    row["Pr(>Chi)"] = (
                        float(chi2.sf(stat, df_diff)) if np.isfinite(stat) else np.nan
                    )
        rows.append(row)
        prev = model

    columns = ["Resid. Df", "Resid. Dev", "Df", "Deviance"] + extra_cols
    return AnovaGAMComparison(
        family_name=family_name,
        test=None if test_name is None else str(test).upper(),
        table=pd.DataFrame(rows, columns=columns),
    )


def anova_gam(
    model,
    *models,
    dispersion: float | None = None,
    test: str | None = None,
    freq: bool = False,
):
    _require_fitted(model)
    if len(models) == 0:
        return _term_table(model, freq=bool(freq), dispersion=dispersion)
    return _comparison_table((model,) + models, test=test, dispersion=dispersion)


__all__ = ["anova_gam", "AnovaGAMSingle", "AnovaGAMComparison"]
