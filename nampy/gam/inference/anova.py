from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2, f

from .._mgcv_constants import LOG_GUARD_MIN
from ..fit import select_covariance_matrix
from ..linalg import (
    mgcv_mroot_chol,
    numerical_rank,
    symmetric_eigh,
    symmetric_eigvalsh,
    symmetrize_matrix,
)
from ..linalg.qr import r_linpack_qr_no_pivot, r_linpack_qr_r
from ..model_state import (
    _coef,
    _coef_full,
    _cov_bayes,
    _cov_unconditional,
    _deviance,
    _edf2,
    _edf_by_term,
    _edf_total,
    _fit_scale,
    _fit_state,
    _fitted_mu,
    _H_coef,
    _require_fitted,
    _summary_R,
    _term_blocks_seq,
    _term_full_coefficient_indices,
)
from .chi_square_mixtures import psum_chisq


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


def _term_fixed(tb) -> bool:
    meta = dict(getattr(tb, "constructor_metadata", {}) or {})
    if "fixed" in meta:
        return bool(meta["fixed"])
    meta = dict(getattr(tb, "metadata", {}) or {})
    if "fixed" in meta:
        return bool(meta["fixed"])
    return False


def _term_null_space_dim(tb) -> int | None:
    for spec in tuple(getattr(tb, "penalty_specs", ()) or ()):
        null_dim = getattr(spec, "null_space_dim", None)
        if null_dim is not None:
            return int(null_dim)
    for meta_name in ("constructor_metadata", "metadata"):
        meta = dict(getattr(tb, meta_name, {}) or {})
        null_dim = meta.get("null_space_dim", None)
        if null_dim is not None:
            return int(null_dim)
    return None


def _term_uses_retest(tb, summary_R) -> bool:
    if summary_R is None or str(getattr(tb, "term_type", "")) == "parametric":
        return False
    if _term_fixed(tb):
        return False
    if str(getattr(tb, "basis_name", "")).lower() in {"fs", "re"}:
        return True
    if any(
        bool(getattr(spec, "is_null_space_penalty", False))
        for spec in tuple(getattr(tb, "penalty_specs", ()) or ())
    ):
        # mgcv/R/smooth.r::smoothCon(null.space.penalty=TRUE) adds the
        # selection penalty and sets smooth$null.space.dim <- 0. summary.gam
        # consequently routes the fully penalized term through reTest().
        return True
    total_penalty = _term_combined_penalty_matrix(tb)
    if total_penalty is None:
        return False
    width = int(total_penalty.shape[0])
    if numerical_rank(total_penalty) >= width:
        return True
    null_dim = _term_null_space_dim(tb)
    if null_dim is not None and null_dim != 0:
        return False
    return numerical_rank(total_penalty) >= width


def _mroot_psd(A: np.ndarray) -> np.ndarray:
    return mgcv_mroot_chol(A)


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

    ind = _term_full_coefficient_indices(model, tb)
    keep = np.setdiff1d(np.arange(q, dtype=int), ind, assume_unique=True)
    perm = np.concatenate([keep, ind])
    LRB = np.asarray(LRB[:, perm], dtype=np.float64)
    packed_qr, _qraux = r_linpack_qr_no_pivot(LRB)
    Rm_full = r_linpack_qr_r(packed_qr)
    block = np.arange(q - ind.size, q, dtype=int)
    Rm = np.asarray(Rm_full[np.ix_(block, block)], dtype=np.float64)

    scale = max(float(_fit_scale(model)), LOG_GUARD_MIN)
    Ve_i = np.asarray(cov_freq[np.ix_(ind, ind)], dtype=np.float64)

    coef_full = np.asarray(_coef_full(model), dtype=np.float64).ravel()
    b_hat = coef_full[ind]
    stat = float(np.sum((Rm @ b_hat) ** 2) / scale)

    B = _mroot_psd(Ve_i)
    if B.shape[1] == 0:
        return 0.0, 0.0, np.nan
    RB = np.asarray(Rm @ B, dtype=np.float64)
    ev = symmetric_eigvalsh((RB.T @ RB) / scale)
    ev = np.clip(np.asarray(ev, dtype=np.float64), 0.0, None)
    tol = (
        max(float(np.max(ev)) if ev.size else 0.0, 0.0)
        * np.finfo(np.float64).eps ** 0.8
    )
    ev = np.asarray(ev[ev > tol], dtype=np.float64)
    rank = int(ev.size)
    if rank <= 0:
        return 0.0, 0.0, np.nan

    if scale_estimated and residual_df > 0.0:
        k = max(1, int(np.round(residual_df)))
        p_value = float(
            psum_chisq(
                0.0,
                np.concatenate([ev, np.array([-stat / k], dtype=np.float64)]),
                df=np.concatenate(
                    [np.ones(rank, dtype=np.int64), np.array([k], dtype=np.int64)]
                ),
            )
        )
    else:
        p_value = float(psum_chisq(stat, ev))
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


def _summary_r_crossprod(model) -> tuple[np.ndarray, float] | None:
    summary_R = _summary_R(model)
    fit_scale = _fit_scale(model)
    if summary_R is None or fit_scale is None:
        return None

    scale = float(fit_scale)
    R = np.asarray(summary_R, dtype=np.float64)
    if (
        R.ndim != 2
        or R.shape[0] != R.shape[1]
        or not np.isfinite(scale)
        or scale <= 0.0
    ):
        return None
    return np.asarray(R.T @ R, dtype=np.float64), scale


def _frequentist_f_matrix(model) -> np.ndarray | None:
    r_info = _summary_r_crossprod(model)
    Vp = _cov_bayes(model)
    if r_info is None or Vp is None:
        return None

    RTR, scale = r_info
    Vp = np.asarray(Vp, dtype=np.float64)
    if Vp.shape != RTR.shape:
        return None
    result: np.ndarray = np.asarray((Vp @ RTR) / scale, dtype=np.float64)
    return result


def _edf1_vector(model) -> np.ndarray:
    H = np.asarray(_H_coef(model), dtype=np.float64)
    if H.ndim == 2 and H.shape[0] == H.shape[1]:
        result: np.ndarray = 2.0 * np.diag(H) - np.sum(H * H.T, axis=1)
        return result

    F = _frequentist_f_matrix(model)
    if F is not None:
        return np.asarray(
            2.0 * np.diag(F) - np.sum(F * F.T, axis=1),
            dtype=np.float64,
        )

    raise ValueError("Model does not expose a square coefficient hat matrix.")


def _term_edf1(model, tb) -> float:
    edf1 = _edf1_vector(model)
    return float(np.sum(edf1[_term_full_coefficient_indices(model, tb)]))


def _approximate_residual_df(model) -> float:
    dfc = 0.0
    edf_total = float(_edf_total(model))
    F = _frequentist_f_matrix(model)
    if F is not None:
        edf_total = float(np.trace(F))
    edf2 = None
    r_info = _summary_r_crossprod(model)
    Vc = _cov_unconditional(model)
    if r_info is not None and Vc is not None:
        RTR, scale = r_info
        Vc = np.asarray(Vc, dtype=np.float64)
        if Vc.shape == RTR.shape:
            edf2 = np.asarray(np.sum(Vc * RTR, axis=1) / scale, dtype=np.float64)
            edf1 = _edf1_vector(model)
            if edf1.shape == edf2.shape and float(np.sum(edf2)) > float(np.sum(edf1)):
                edf2 = np.asarray(edf1, dtype=np.float64).copy()
    if edf2 is None:
        stored = _edf2(model)
        if stored is not None:
            edf2 = np.asarray(stored, dtype=np.float64)
    if edf2 is not None:
        dfc = float(np.sum(edf2)) - edf_total
    return float(model.n_samples_) - float(np.sum(_edf1_vector(model))) - dfc


def _comparison_residual_deviance(model) -> float:
    family_class = str(getattr(model.family, "family_class", "")).lower()
    if family_class in {"extended", "general"}:
        return float(_deviance(model))

    y = np.asarray(model.y_, dtype=np.float64)
    mu = _fitted_mu(model)
    if mu is None:
        return float(_deviance(model))

    prior_weights = model.prior_weights_
    weights = None if prior_weights is None else np.asarray(prior_weights, dtype=np.float64)
    try:
        return float(
            model.family.deviance(
                y,
                np.asarray(mu, dtype=np.float64),
                weights=weights,
            )
        )
    except Exception:
        return float(_deviance(model))


def _stable_wald_stat(beta: np.ndarray, cov: np.ndarray) -> tuple[float, int]:
    beta = np.asarray(beta, dtype=np.float64).ravel()
    cov = np.asarray(cov, dtype=np.float64)
    if beta.size == 0:
        return 0.0, 0

    if cov.shape != (beta.size, beta.size):
        raise ValueError(
            "Coefficient covariance block shape does not match term width."
        )

    rank = numerical_rank(cov)
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

    # Mirror mgcv/R/mgcv.r::testStat(): qrx <- qr(X, tol=0). This is base R's
    # non-LAPACK dqrdc2 Householder path; at tol=0 its limited near-zero-column
    # pivot is disabled. A LAPACK QR spans the same space but is not an
    # equivalent representation for this ill-conditioned statistic.
    packed_qr, _qraux = r_linpack_qr_no_pivot(X)
    R = r_linpack_qr_r(packed_qr)
    Vt = R @ V @ R.T
    Vt = symmetrize_matrix(Vt)
    evals, evecs = symmetric_eigh(Vt, descending=True)
    # mgcv/R/mgcv.r::testStat(): "remove possible ambiguity from statistic".
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
                    psum_chisq(
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
                    psum_chisq(
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
            pval = 0.5 * (float(psum_chisq(d, val)) + float(psum_chisq(d1, val)))
    else:
        pval = 2.0

    if pval > 1.0:
        if residual_df > 0.0 and rank1 > 0.0:
            pval = 0.5 * (
                float(f.sf(d / rank1, rank1, residual_df))
                + float(f.sf(d1 / rank1, rank1, residual_df))
            )
        else:
            pval = 0.5 * (float(chi2.sf(d, rank1)) + float(chi2.sf(d1, rank1)))
    # mgcv/R/mgcv.r::testStat() only caps the upper tail.  In particular, it
    # preserves the tiny negative qfc artifacts that can occur at the edge of
    # its numerical tolerance instead of silently changing the reported value.
    return d, rank1, min(pval, 1.0)


def _term_table(
    model, *, freq: bool, dispersion: float | None, re_test: bool = True
) -> AnovaGAMSingle:
    scale_est = _scale_estimated(model)
    resid_df = _residual_df(model)
    fit_scale = float(_fit_scale(model))
    disp = float(fit_scale if dispersion is None else dispersion)

    V_para = select_covariance_matrix(model, cov=("freq" if freq else "bayes"))
    V_smooth = select_covariance_matrix(model, cov="bayes")
    try:
        V_freq = select_covariance_matrix(model, cov="freq")
    except Exception:
        V_freq = None

    if dispersion is not None:
        # mgcv/R/mgcv.r:3895-3900: a supplied dispersion rescales every
        # covariance through covmat.unscaled and forces est.disp <- FALSE
        # (z / Chi.sq columns).
        factor = disp / fit_scale
        scale_est = False
        V_para = None if V_para is None else np.asarray(V_para, np.float64) * factor
        V_smooth = (
            None if V_smooth is None else np.asarray(V_smooth, np.float64) * factor
        )
        V_freq = None if V_freq is None else np.asarray(V_freq, np.float64) * factor

    beta = np.asarray(_coef(model), dtype=np.float64).ravel()
    edf_by_term = np.asarray(_edf_by_term(model), dtype=np.float64).ravel()
    summary_R = _summary_R(model)
    fit_state = _fit_state(model)

    param_rows: list[dict[str, object]] = []
    smooth_rows: list[dict[str, object]] = []

    for group in _parametric_term_groups(model):
        beta_cols: list[int] = []
        full_cols: list[int] = []
        for tb in group["blocks"]:
            beta_cols.extend(range(tb.coef_slice.start, tb.coef_slice.stop))
            full_cols.extend(_term_full_coefficient_indices(model, tb).tolist())

        beta_idx = np.asarray(beta_cols, dtype=np.int64)
        full_idx = np.asarray(full_cols, dtype=np.int64)
        beta_i = np.asarray(beta[beta_idx], dtype=np.float64)
        cov_i = (
            None
            if V_para is None
            else np.asarray(V_para[np.ix_(full_idx, full_idx)], dtype=np.float64)
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
        full_idx = _term_full_coefficient_indices(model, tb)
        beta_i = beta[sl]
        edf_i = float(edf_by_term[i]) if i < edf_by_term.size else float(beta_i.size)

        cov_i = (
            None
            if V_smooth is None
            else np.asarray(V_smooth[np.ix_(full_idx, full_idx)], dtype=np.float64)
        )
        if cov_i is None:
            stat, ref_df, p_value = np.nan, max(edf_i, 1.0), np.nan
        elif _term_uses_retest(tb, summary_R):
            if not re_test:
                # mgcv/R/mgcv.r:4021-4022: with re.test=FALSE the reTest-eligible
                # smooth produces no row in the s.table.
                continue
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
            X_i = np.asarray(
                (
                    summary_R[:, full_idx]
                    if summary_R is not None
                    else fit_state.X[:, full_idx]
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
    if test_name not in {None, "chisq", "f"}:
        raise ValueError("test must be one of None, 'Chisq', 'LRT', or 'F'.")

    family_class = str(getattr(models[0].family, "family_class", "")).lower()
    extended_like = family_class == "extended"
    use_loglik_deviance = extended_like or family_class == "general"
    resid_dfs = [float(_approximate_residual_df(model)) for model in models]
    # mgcv::anova.gam (mgcv/R/mgcv.r:4148) delegates to stats::anova.glmlist,
    # which takes the reference dispersion from the model with the smallest
    # residual df ("bigmodel"); mgcv.r:4136-4140 uses the same model for
    # extended families.
    big_index = int(np.argmin(np.asarray(resid_dfs, dtype=np.float64)))
    if dispersion is None:
        disp_obj = _fit_scale(models[big_index])
        disp = 1.0 if disp_obj is None else float(disp_obj)
    else:
        disp = float(dispersion)
    if not np.isfinite(disp) or disp <= 0.0:
        disp = 1.0
    # stats::anova.glmlist (R >= 4.4): df.dispersion is the residual df of the
    # biggest model for estimated-dispersion families (family$dispersion NA),
    # Inf for known-dispersion families, and falls back to the dispersion==1
    # check when the family carries no dispersion field (mgcv extended and
    # general families).
    known_scale = getattr(models[0].family, "known_scale", None)
    if use_loglik_deviance:
        df_scale = np.inf if disp == 1.0 else float(min(resid_dfs))
    elif known_scale is None:
        df_scale = float(min(resid_dfs))
    else:
        df_scale = np.inf
    chosen_test = test_name
    if test is None:
        # stats::anova.glmlist (R >= 4.4) applies a family-based default test
        # when test is NULL: "Chisq" with an explicit dispersion, "F" for
        # estimated-dispersion families, "Chisq" for known-dispersion ones,
        # and no test when family$dispersion is absent (extended/general).
        if dispersion is not None:
            chosen_test = "chisq"
        elif use_loglik_deviance:
            chosen_test = None
        elif known_scale is None:
            chosen_test = "f"
        else:
            chosen_test = "chisq"
    if extended_like and chosen_test is not None:
        chosen_test = "chisq"
    extra_cols: list[str] = []
    if chosen_test == "f":
        extra_cols = ["F", "Pr(>F)"]
    elif chosen_test == "chisq":
        extra_cols = ["Pr(>Chi)"]
    rows: list[dict[str, object]] = []
    prev = None
    prev_resid_df = np.nan
    prev_dev = np.nan
    for idx, model in enumerate(models):
        resid_df = resid_dfs[idx]
        resid_dev = float(_comparison_residual_deviance(model))
        if use_loglik_deviance:
            resid_dev = float(-2.0 * float(model.loglik()) * disp)
        row: dict[str, object] = {
            "Resid. Df": resid_df,
            "Resid. Dev": resid_dev,
            "Df": np.nan,
            "Deviance": np.nan,
        }
        for col in extra_cols:
            row[col] = np.nan

        if prev is not None:
            df_diff = prev_resid_df - resid_df
            dev_diff = prev_dev - resid_dev
            row["Df"] = df_diff
            row["Deviance"] = dev_diff

            if chosen_test == "f":
                # stats::stat.anova "F": Fvalue <- (Deviance/Df)/scale, NA when
                # Df == 0 or the statistic is negative; p = pf(F, |Df|,
                # df.scale) with the chi-square limit at df.scale == Inf.
                if df_diff == 0.0 or not np.isfinite(df_diff):
                    stat = np.nan
                else:
                    stat = float((dev_diff / df_diff) / disp)
                    if stat < 0.0:
                        stat = np.nan
                row["F"] = stat
                if np.isfinite(stat):
                    if np.isinf(df_scale):
                        row["Pr(>F)"] = float(
                            chi2.sf(stat * abs(df_diff), abs(df_diff))
                        )
                    else:
                        row["Pr(>F)"] = float(f.sf(stat, abs(df_diff), df_scale))
            elif chosen_test == "chisq":
                # stats::stat.anova "Chisq": vals <- Deviance/scale * sign(Df),
                # NA when Df == 0 or vals < 0; p = pchisq(vals, |Df|).
                if df_diff == 0.0 or not np.isfinite(df_diff):
                    stat = np.nan
                else:
                    stat = float(dev_diff / disp) * float(np.sign(df_diff))
                    if stat < 0.0:
                        stat = np.nan
                row["Pr(>Chi)"] = (
                    float(chi2.sf(stat, abs(df_diff))) if np.isfinite(stat) else np.nan
                )
        rows.append(row)
        prev = model
        prev_resid_df = resid_df
        prev_dev = resid_dev

    columns = ["Resid. Df", "Resid. Dev", "Df", "Deviance"] + extra_cols
    return AnovaGAMComparison(
        family_name=family_name,
        test=None if chosen_test is None else str(chosen_test).upper(),
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
