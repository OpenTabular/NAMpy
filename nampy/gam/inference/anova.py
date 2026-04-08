from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.stats import chi2, f

from ..fit.covariance import select_covariance_matrix


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


def _require_fitted(model):
    if not getattr(model, "_fitted", False):
        raise RuntimeError("Model is not fitted.")


def _residual_df(model) -> float:
    intercept_df = 1.0 if bool(getattr(model, "fit_intercept", False)) else 0.0
    return float(model.n_samples_) - intercept_df - float(model.edf_)


def _edf1_vector(model) -> np.ndarray:
    H = np.asarray(getattr(model, "_H_coef", None), dtype=np.float64)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("Model does not expose a square coefficient hat matrix.")
    return 2.0 * np.diag(H) - np.sum(H * H.T, axis=1)


def _x_col_offset(model) -> int:
    """Number of columns prepended to X before the coef_ columns (intercept)."""
    return 1 if bool(getattr(model, "fit_intercept", False)) else 0


def _term_edf1(model, tb) -> float:
    sl = tb.coef_slice
    offset = _x_col_offset(model)
    edf1 = _edf1_vector(model)
    return float(np.sum(edf1[sl.start + offset : sl.stop + offset]))


def _residual_df_approx_mgcv(model) -> float:
    intercept_df = 1.0 if bool(getattr(model, "fit_intercept", False)) else 0.0
    return float(model.n_samples_) - intercept_df - float(np.sum(_edf1_vector(model)))


def _stable_wald_stat(beta: np.ndarray, cov: np.ndarray) -> tuple[float, int]:
    beta = np.asarray(beta, dtype=np.float64).ravel()
    cov = np.asarray(cov, dtype=np.float64)
    if beta.size == 0:
        return 0.0, 0

    if cov.shape != (beta.size, beta.size):
        raise ValueError("Coefficient covariance block shape does not match term width.")

    rank = int(np.linalg.matrix_rank(cov))
    if rank == 0:
        return 0.0, 0

    cov_pinv = np.linalg.pinv(cov, hermitian=True)
    stat = float(beta @ cov_pinv @ beta)
    return max(stat, 0.0), rank


def _wald_p_value(stat: float, ref_df: float, residual_df: float, *, gaussian: bool) -> tuple[str, float]:
    if not np.isfinite(stat) or not np.isfinite(ref_df) or ref_df <= 0.0:
        return ("F" if gaussian else "ChiSq"), np.nan

    if gaussian and np.isfinite(residual_df) and residual_df > 0.0:
        f_stat = float(stat / ref_df)
        return "F", float(f.sf(f_stat, ref_df, residual_df))

    return "ChiSq", float(chi2.sf(stat, ref_df))


def _smooth_test_stat(p: np.ndarray, X: np.ndarray, V: np.ndarray, rank: float, residual_df: float) -> tuple[float, float, float]:
    X = np.asarray(X, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64).ravel()
    if X.ndim != 2 or V.shape != (X.shape[1], X.shape[1]) or p.size != X.shape[1]:
        raise ValueError("Smooth test inputs have inconsistent shapes.")

    # mgcv::testStat() uses a pivoted QR and permutes V/p into that basis.
    _, R, pivot = qr(X, mode="economic", pivoting=True)
    p = p[np.asarray(pivot, dtype=np.intp)]
    V = V[np.ix_(pivot, pivot)]
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

    tol = max(float(np.max(np.abs(evals))) if evals.size else 0.0, 1.0) * np.finfo(np.float64).eps ** 0.9
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
            vec[:, : (k - 1)] = vec[:, : (k - 1)] / np.sqrt(np.clip(evals[: (k - 1)], 1e-300, None))
        b12 = np.sqrt(max(0.5 * nu * (1.0 - nu), 0.0))
        B = np.array([[1.0, b12], [b12, nu]], dtype=np.float64)
        ev = np.diag(np.power(np.clip(evals[(k - 1) : k1], 1e-300, None), -0.5))
        B = ev @ B @ ev
        eb_vals, eb_vecs = np.linalg.eigh(B)
        rB = eb_vecs @ np.diag(np.sqrt(np.clip(eb_vals, 0.0, None))) @ eb_vecs.T
        vec1 = vec.copy()
        vec1[:, (k - 1) : k1] = (rB @ np.diag([-1.0, 1.0]) @ vec[:, (k - 1) : k1].T).T
        vec[:, (k - 1) : k1] = (rB @ vec[:, (k - 1) : k1].T).T
    else:
        scale = np.sqrt(np.clip(evals[:k1], 1e-300, None))
        vec = vec / scale
        vec1 = vec.copy()
        if k == 1:
            rank = 1.0

    Rp = R @ p
    d = float(np.sum((vec.T @ Rp) ** 2))
    d1 = float(np.sum((vec1.T @ Rp) ** 2))
    ref_df = 1.0 if nu > 0.0 and k1 == 1 else float(rank)
    if residual_df > 0.0 and ref_df > 0.0:
        pval = 0.5 * (
            float(f.sf(d / ref_df, ref_df, residual_df))
            + float(f.sf(d1 / ref_df, ref_df, residual_df))
        )
    else:
        pval = 0.5 * (
            float(chi2.sf(d, ref_df))
            + float(chi2.sf(d1, ref_df))
        )
    return d, ref_df, min(max(pval, 0.0), 1.0)


def _term_table(model, *, freq: bool, dispersion: float | None) -> AnovaGAMSingle:
    gaussian = str(getattr(model.family, "name", "")).lower() == "gaussian"
    resid_df = _residual_df(model)
    disp = float(model.scale_ if dispersion is None else dispersion)

    V_para = select_covariance_matrix(model, cov=("freq" if freq else "bayes"))
    V_smooth = select_covariance_matrix(model, cov="bayes")

    beta = np.asarray(model.coef_, dtype=np.float64).ravel()
    edf_by_term = np.asarray(model.edf_by_term_, dtype=np.float64).ravel()
    summary_R = getattr(model, "_summary_R_", None)

    param_rows: list[dict[str, object]] = []
    smooth_rows: list[dict[str, object]] = []

    x_offset = _x_col_offset(model)

    for i, tb in enumerate(getattr(model, "term_blocks_", ()) or ()):
        sl = tb.coef_slice
        # sl indexes coef_ (no intercept); Vp_/Vf_ include the intercept column so
        # we shift by x_offset when extracting covariance submatrices.
        x_sl = slice(sl.start + x_offset, sl.stop + x_offset)
        beta_i = beta[sl]
        edf_i = float(edf_by_term[i]) if i < edf_by_term.size else float(beta_i.size)

        if str(getattr(tb, "term_type", "")) == "parametric":
            cov_i = None if V_para is None else np.asarray(V_para[x_sl, x_sl], dtype=np.float64)
            stat, rank = (np.nan, int(beta_i.size)) if cov_i is None else _stable_wald_stat(beta_i, cov_i)
            ref_df = float(rank)
            test_name, p_value = _wald_p_value(stat, ref_df, resid_df, gaussian=gaussian)
            stat_out = float(stat / ref_df) if (gaussian and np.isfinite(stat) and ref_df > 0.0) else float(stat)
            param_rows.append(
                {
                    "label": str(tb.label),
                    "df": ref_df,
                    "wald_stat": stat_out,
                    "p_value": p_value,
                    "test": test_name,
                    "covariance": "freq" if freq else "bayes",
                    "dispersion": disp,
                }
            )
            continue

        cov_i = None if V_smooth is None else np.asarray(V_smooth[x_sl, x_sl], dtype=np.float64)
        if cov_i is None:
            stat, ref_df, p_value = np.nan, max(edf_i, 1.0), np.nan
        else:
            x_start = int(x_sl.start)
            x_stop = int(x_sl.stop)
            X_i = np.asarray(
                (
                    summary_R[:, x_start:x_stop]
                    if summary_R is not None
                    else model.fit_state_.X[:, x_start:x_stop]
                ),
                dtype=np.float64,
            )
            edf1_i = _term_edf1(model, tb)
            stat, ref_df, p_value = _smooth_test_stat(
                beta_i,
                X_i,
                cov_i,
                rank=min(float(X_i.shape[1]), max(edf1_i, 1.0)),
                residual_df=(resid_df if gaussian else -1.0),
            )
        test_name = "F" if gaussian else "ChiSq"
        stat_out = float(stat / ref_df) if (gaussian and np.isfinite(stat) and ref_df > 0.0) else float(stat)
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
            columns=["label", "df", "wald_stat", "p_value", "test", "covariance", "dispersion"],
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


def _comparison_table(models: tuple, *, test: str | None, dispersion: float | None) -> AnovaGAMComparison:
    family_name = str(models[0].family.name)
    method_name = str(getattr(models[0], "_optim_method", "")).lower()
    n_samples = int(models[0].n_samples_)

    for model in models:
        _require_fitted(model)
        if str(model.family.name) != family_name:
            raise ValueError("anova.gam multi-model comparisons require the same family.")
        if int(model.n_samples_) != n_samples:
            raise ValueError("anova.gam multi-model comparisons require the same sample size.")
        if str(getattr(model, "_optim_method", "")).lower() != method_name:
            raise ValueError(
                "anova.gam multi-model comparisons require the same smoothing selection method."
            )

    test_name = None if test is None else str(test).strip().lower()
    if test_name not in {None, "chisq", "f", "cp"}:
        raise ValueError("test must be one of None, 'Chisq', 'F', or 'Cp'.")

    disp = float(models[-1].scale_ if dispersion is None else dispersion)
    gaussian = family_name.lower() == "gaussian"

    rows: list[dict[str, object]] = []
    prev = None
    for idx, model in enumerate(models):
        resid_df = _residual_df_approx_mgcv(model)
        row = {
            "model": idx,
            "formula": getattr(model, "formula_", None) or getattr(model, "formula", None),
            "edf": float(model.edf_),
            "residual_df": resid_df,
            "deviance": float(model.deviance_),
            "criterion": getattr(model, "smoothing_score_", None),
            "edf_diff": np.nan,
            "deviance_diff": np.nan,
            "statistic": np.nan,
            "p_value": np.nan,
            "test": None if test_name is None else str(test).upper(),
        }

        if prev is not None:
            edf_diff = float(model.edf_) - float(prev.edf_)
            dev_diff = float(prev.deviance_) - float(model.deviance_)
            row["edf_diff"] = edf_diff
            row["deviance_diff"] = dev_diff

            if edf_diff > 0.0 and dev_diff >= 0.0:
                chosen = "f" if (test_name == "f" or (test_name is None and gaussian)) else test_name
                if chosen == "cp":
                    row["statistic"] = float(
                        (float(model.smoothing_score_) if model.smoothing_score_ is not None else model.deviance_)
                        - (float(prev.smoothing_score_) if prev.smoothing_score_ is not None else prev.deviance_)
                    )
                    row["test"] = "CP"
                elif chosen == "f":
                    denom_df = max(float(model.n_samples_) - float(model.edf_) - (1.0 if model.fit_intercept else 0.0), 1.0)
                    denom = float(model.deviance_) / denom_df
                    stat = np.nan if denom <= 0.0 else float((dev_diff / edf_diff) / denom)
                    row["statistic"] = stat
                    row["p_value"] = float(f.sf(stat, edf_diff, denom_df)) if np.isfinite(stat) else np.nan
                    row["test"] = "F"
                else:
                    stat = float(dev_diff / disp) if disp > 0.0 else np.nan
                    row["statistic"] = stat
                    row["p_value"] = float(chi2.sf(stat, edf_diff)) if np.isfinite(stat) else np.nan
                    row["test"] = "CHISQ"
        rows.append(row)
        prev = model

    return AnovaGAMComparison(
        family_name=family_name,
        test=None if test_name is None else str(test).upper(),
        table=pd.DataFrame(
            rows,
            columns=[
                "model",
                "formula",
                "edf",
                "residual_df",
                "deviance",
                "criterion",
                "edf_diff",
                "deviance_diff",
                "statistic",
                "p_value",
                "test",
            ],
        ),
    )


def anova_gam(model, *models, dispersion: float | None = None, test: str | None = None, freq: bool = False):
    _require_fitted(model)
    if len(models) == 0:
        return _term_table(model, freq=bool(freq), dispersion=dispersion)
    return _comparison_table((model,) + models, test=test, dispersion=dispersion)


__all__ = ["anova_gam", "AnovaGAMSingle", "AnovaGAMComparison"]
