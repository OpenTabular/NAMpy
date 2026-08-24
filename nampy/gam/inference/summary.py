"""mgcv ``summary.gam`` port.

Upstream spec: ``mgcv/R/mgcv.r:3858-4068`` (``summary.gam``). The parametric
term table and the smooth significance table reuse the existing
``anova``/``testStat`` port (:func:`nampy.gam.inference.anova._term_table`);
this module adds the per-coefficient ``p.table``, ``r.sq``, ``dev.expl``,
and the remaining summary scalars.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t

from ..fit import select_covariance_matrix
from ..model_state import (
    _coef_column_offset,
    _coef_full,
    _deviance,
    _edf_total,
    _fit_scale,
    _fit_state,
    _fitted_mu,
    _predictor_designs,
    _predictor_full_slices,
    _require_fitted,
    _term_blocks_seq,
    _term_full_coefficient_indices,
)
from ..term_labels import mgcv_term_display_label
from .anova import (
    _parametric_term_groups,
    _residual_df,
    _scale_estimated,
    _term_table,
)
from .null_deviance import null_deviance


@dataclass(frozen=True)
class GAMSummary:
    """Structured mirror of mgcv's ``summary.gam`` return list."""

    family_name: str
    link_name: str
    formula: Any
    p_coeff: np.ndarray
    se: np.ndarray
    p_t: np.ndarray
    p_pv: np.ndarray
    p_table: pd.DataFrame
    pterms_table: pd.DataFrame
    s_table: pd.DataFrame
    residual_df: float
    m: int
    scale: float
    dispersion: float
    scale_estimated: bool
    r_sq: float | None
    dev_expl: float | None
    null_deviance: float | None
    deviance: float
    n: int
    np: int
    rank: int | None
    method: str | None
    sp_criterion: float | None
    covariance: str
    edf_total: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        from ..diagnostics.summary_format import summary_lines_from_gam_summary

        return "\n".join(summary_lines_from_gam_summary(self))


_METHOD_DISPLAY = {
    "reml": "-REML",
    "laml": "-REML",
    "ml": "-ML",
    "gcv": "GCV",
    "ubre": "UBRE",
    "aic": "UBRE",
    "ubreaic": "UBRE",
}


def _parametric_coefficient_indices(model) -> tuple[list[int], list[str]]:
    """
    Mirror mgcv/R/mgcv.r:3907-3913: the p.table rows are the intercept(s)
    plus the strictly parametric coefficients, in coefficient order.
    Names are synthesized from term labels (display only).
    """
    indices: list[int] = []
    names: list[str] = []

    predictor_slices = list(_predictor_full_slices(model) or [])
    if len(predictor_slices) > 1:
        for k, (pred, sl) in enumerate(
            zip(_predictor_designs(model), predictor_slices, strict=True)
        ):
            if bool(getattr(pred, "has_intercept", False)):
                indices.append(int(sl.start))
                names.append("(Intercept)" if k == 0 else f"(Intercept).{k}")
    else:
        x_offset = int(_coef_column_offset(model))
        if x_offset >= 1 and bool(getattr(model, "fit_intercept", True)):
            indices.append(0)
            names.append("(Intercept)")

    for group in _parametric_term_groups(model):
        for tb in group["blocks"]:
            full_indices = _term_full_coefficient_indices(model, tb)
            term_label = mgcv_term_display_label(tb)
            for j, full_index in enumerate(full_indices):
                indices.append(int(full_index))
                names.append(
                    term_label
                    if full_indices.size == 1
                    else f"{term_label}.{j}"
                )
    return indices, names


def _r_squared_adjusted(model, residual_df: float) -> float | None:
    """Mirror mgcv/R/mgcv.r:4051-4056 (R ``var()`` semantics, sqrt weights)."""
    family_class = str(getattr(model.family, "family_class", "")).lower()
    if family_class == "general" or bool(getattr(model.family, "no_r_sq", False)):
        return None
    y = np.asarray(model.y_, dtype=np.float64)
    fitted = np.asarray(_fitted_mu(model), dtype=np.float64).ravel()
    w = (
        np.ones_like(y)
        if model.prior_weights_ is None
        else np.asarray(model.prior_weights_, dtype=np.float64)
    )
    mean_y = float(np.sum(w * y) / np.sum(w))
    ws = np.sqrt(w)
    nobs = float(len(y))
    if nobs < 2.0 or residual_df <= 0.0:
        return None
    v_resid = float(np.var(ws * (y - fitted), ddof=1))
    v_null = float(np.var(ws * (y - mean_y), ddof=1))
    if not np.isfinite(v_null) or v_null <= 0.0:
        return None
    return float(1.0 - v_resid * (nobs - 1.0) / (v_null * residual_df))


def _summary_deviance(model) -> float:
    dev = float(_deviance(model))
    if str(getattr(model.family, "family_class", "")).lower() == "general":
        # Mirror mgcv/R/mgcv.r::gam post-fit fill: general-family deviance is
        # the sum of squared deviance residuals.
        rsd = np.asarray(model.residuals(type="deviance"), dtype=np.float64)
        dev_res = float(np.sum(rsd**2.0))
        if np.isfinite(dev_res):
            return dev_res
    return dev


def summary_gam(
    model,
    *,
    dispersion: float | None = None,
    freq: bool = False,
    re_test: bool = True,
) -> GAMSummary:
    """Build the mgcv ``summary.gam`` object for a fitted model."""
    _require_fitted(model)

    scale_est = _scale_estimated(model)
    resid_df = float(_residual_df(model))
    fit_scale = float(_fit_scale(model))
    disp = float(fit_scale if dispersion is None else dispersion)

    covmat = select_covariance_matrix(model, cov=("freq" if freq else "bayes"))
    if covmat is not None and dispersion is not None:
        # mgcv/R/mgcv.r:3895-3900.
        covmat = np.asarray(covmat, dtype=np.float64) * (disp / fit_scale)
    if dispersion is not None:
        scale_est = False

    coef = np.asarray(_coef_full(model), dtype=np.float64).ravel()
    se_full = (
        np.full_like(coef, np.nan)
        if covmat is None
        else np.sqrt(np.clip(np.diag(np.asarray(covmat, dtype=np.float64)), 0.0, None))
    )

    ind, names = _parametric_coefficient_indices(model)
    p_coeff = coef[ind] if ind else np.empty(0, dtype=np.float64)
    p_se = se_full[ind] if ind else np.empty(0, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_t = np.where(p_se > 0.0, p_coeff / p_se, np.nan)
    if scale_est:
        # mgcv/R/mgcv.r:3921-3925.
        p_pv = 2.0 * student_t.sf(np.abs(p_t), df=resid_df)
        stat_col, p_col = "t value", "Pr(>|t|)"
    else:
        # mgcv/R/mgcv.r:3917-3920.
        p_pv = 2.0 * norm.sf(np.abs(p_t))
        stat_col, p_col = "z value", "Pr(>|z|)"
    p_table = pd.DataFrame(
        {
            "Estimate": p_coeff,
            "Std. Error": p_se,
            stat_col: p_t,
            p_col: p_pv,
        },
        index=names,
    )

    tables = _term_table(model, freq=freq, dispersion=dispersion, re_test=re_test)

    dev = _summary_deviance(model)
    try:
        null_dev = float(null_deviance(model))
    except NotImplementedError:
        null_dev = None
    dev_expl = (
        None
        if null_dev is None or not np.isfinite(null_dev) or null_dev == 0.0
        else float((null_dev - dev) / null_dev)
    )

    r_sq = _r_squared_adjusted(model, resid_df)

    method_key = str(getattr(model, "_optim_method", "") or "").lower()
    method = _METHOD_DISPLAY.get(method_key)
    sp_criterion = (
        None
        if getattr(model, "smoothing_score_", None) is None
        else float(model.smoothing_score_)
    )

    fit_state = _fit_state(model)
    rank = getattr(fit_state, "penalized_system_rank", None)
    rank = None if rank is None else int(rank)

    n_smooth = len(
        [
            tb
            for tb in _term_blocks_seq(model)
            if str(getattr(tb, "term_type", "")) != "parametric"
        ]
    )

    edf_total = float(_edf_total(model))

    return GAMSummary(
        family_name=str(model.family.name),
        link_name=str(model.family.link_name),
        formula=getattr(model, "formula", None),
        p_coeff=p_coeff,
        se=se_full,
        p_t=np.asarray(p_t, dtype=np.float64),
        p_pv=np.asarray(p_pv, dtype=np.float64),
        p_table=p_table,
        pterms_table=tables.parametric_table,
        s_table=tables.smooth_table,
        residual_df=resid_df,
        m=int(n_smooth),
        scale=disp,
        dispersion=disp,
        scale_estimated=bool(scale_est),
        r_sq=r_sq,
        dev_expl=dev_expl,
        null_deviance=null_dev,
        deviance=float(dev),
        n=int(model.n_samples_),
        np=int(coef.size),
        rank=rank,
        method=method,
        sp_criterion=sp_criterion,
        covariance="freq" if freq else "bayes",
        edf_total=edf_total,
    )


__all__ = ["GAMSummary", "summary_gam"]
