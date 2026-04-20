"""
Post-fit model summary generation.

:func:`build_summary_lines` produces a human-readable tabular summary of a
fitted GAM: family, link, effective degrees of freedom by term, smoothing
parameters, and goodness-of-fit statistics.
"""

from __future__ import annotations

from .._model_state import (
    _deviance,
    _edf_by_term,
    _edf_total,
    _fit_intercept,
    _fit_scale,
    _fit_summary,
    _intercept,
    _require_fitted,
    _rss,
    _term_blocks_seq,
)

TERM_COL_WIDTH = 20
EDF_COL_WIDTH = 8
K_COL_WIDTH = 5
SP_COL_WIDTH = 20
SUMMARY_WIDTH = TERM_COL_WIDTH + EDF_COL_WIDTH + K_COL_WIDTH + SP_COL_WIDTH + 3


def _format_term_cell(tb) -> str:
    prefix = _term_prefix(tb)
    return f"{prefix}({tb.label})"


def _format_summary_row(term: str, edf: float, k: int, sp_txt: str) -> str:
    return (
        f"{term:<{TERM_COL_WIDTH}s} "
        f"{edf:>{EDF_COL_WIDTH}.3f} "
        f"{k:>{K_COL_WIDTH}d} "
        f"{sp_txt:>{SP_COL_WIDTH}s}"
    )


def _term_prefix(tb) -> str:
    if tb.basis_name == "mrf":
        return "mrf"
    if tb.term_type == "random_effect":
        return "re"
    if tb.term_type == "parametric":
        return "p"
    return "s"


def _format_sp_values(values) -> str:
    if len(values) == 0:
        return "NA"
    if len(values) == 1:
        return f"{values[0]:.4g}"
    return "[" + ", ".join(f"{v:.3g}" for v in values) + "]"


def build_summary_lines(model) -> list[str]:
    _require_fitted(model)
    fit_summary = _fit_summary(model)
    term_blocks = _term_blocks_seq(model)

    lines = []
    lines.append("General Smooth Model Summary")
    lines.append("=" * SUMMARY_WIDTH)
    lines.append(f"Family : {model.family.name}")
    lines.append(f"Link : {model.family.link_name}")
    lines.append(f"Smoothing method : {model._optim_method}")
    lines.append(f"n samples : {model.n_samples_}")
    if model.smoothing_score_ is not None:
        lines.append(f"Smoothing score : {model.smoothing_score_:.6g}")
    if model.offset_train_ is not None:
        lines.append("Offset : yes")
    lines.append("")

    lines.append(
        f"{'term':<{TERM_COL_WIDTH}s} "
        f"{'edf':>{EDF_COL_WIDTH}s} "
        f"{'k':>{K_COL_WIDTH}s} "
        f"{'sp':>{SP_COL_WIDTH}s}"
    )
    lines.append("-" * SUMMARY_WIDTH)

    for i, tb in enumerate(term_blocks):
        k_i = tb.coef_slice.stop - tb.coef_slice.start
        sp_vals = [model.smoothing_params[j] for j in tb.smoothing_indices]
        sp_txt = _format_sp_values(sp_vals)
        lines.append(
            _format_summary_row(
                _format_term_cell(tb),
                (
                    fit_summary.edf_by_term[i]
                    if fit_summary is not None
                    else _edf_by_term(model)[i]
                ),
                k_i,
                sp_txt,
            )
        )

    lines.append("-" * SUMMARY_WIDTH)
    if _fit_intercept(model):
        intercept = (
            fit_summary.intercept if fit_summary is not None else _intercept(model)
        )
        lines.append(f"Intercept : {intercept:.6g}")
    edf_total = fit_summary.edf_total if fit_summary is not None else _edf_total(model)
    scale = fit_summary.scale if fit_summary is not None else _fit_scale(model)
    lines.append(f"EDF (total) : {edf_total:.3f}")
    lines.append(f"Scale : {scale:.6g}")
    if model.family.name == "gaussian":
        rss = fit_summary.rss if fit_summary is not None else _rss(model)
        lines.append(f"RSS : {rss:.6g}")
    else:
        deviance = fit_summary.deviance if fit_summary is not None else _deviance(model)
        lines.append(f"Deviance : {deviance:.6g}")

    return lines


def summary_text(model) -> str:
    return "\n".join(build_summary_lines(model))


def print_summary(model) -> str:
    text = summary_text(model)
    print(text)
    return text


__all__ = ["build_summary_lines", "summary_text", "print_summary"]
