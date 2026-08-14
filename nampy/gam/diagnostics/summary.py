"""
mgcv ``print.summary.gam``-shaped rendering of a fitted GAM summary.

Layout mirrors ``mgcv/R/mgcv.r:4070-4099``: family/link lines, formula echo,
the parametric coefficient table, the smooth significance table, an optional
``Rank`` line, ``R-sq.(adj)`` / ``Deviance explained``, and the trailing
``<method> = <score>  Scale est. = <scale>  n = <n>`` line. Exact column
widths are not a parity surface; the numbers are (see
:mod:`nampy.gam.inference.summary`).
"""

from __future__ import annotations

import numpy as np


def _fmt_g(value, digits=5) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}g}"


def _p_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return " "


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 2e-16:
        return "<2e-16"
    return f"{p:.4g}"


def _coefmat_lines(df, *, p_col: str) -> list[str]:
    if df is None or len(df) == 0:
        return []
    labels = [str(v) for v in df.index]
    columns = list(df.columns)
    label_w = max(12, max(len(v) for v in labels) + 1)
    header = " " * label_w + "".join(f"{c:>13s}" for c in columns) + "    "
    lines = [header]
    for label, (_, row) in zip(labels, df.iterrows(), strict=True):
        cells = []
        p_value = np.nan
        for c in columns:
            v = row[c]
            if c == p_col:
                p_value = float(v) if v is not None else np.nan
                cells.append(f"{_fmt_p(p_value):>13s}")
            else:
                cells.append(f"{_fmt_g(float(v)):>13s}")
        lines.append(f"{label:<{label_w}s}" + "".join(cells) + f" {_p_stars(p_value)}")
    lines.append("---")
    lines.append(
        "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
    )
    return lines


def _formula_lines(formula) -> list[str]:
    if formula is None:
        return []
    lines = ["Formula:"]
    if isinstance(formula, (list, tuple)):
        lines.extend(str(f) for f in formula)
    else:
        lines.append(str(formula))
    return lines


def summary_lines_from_gam_summary(summary) -> list[str]:
    """Render a :class:`~nampy.gam.inference.summary.GAMSummary`."""
    lines: list[str] = []
    lines.append("")
    lines.append(f"Family: {summary.family_name}")
    lines.append(f"Link function: {summary.link_name}")
    lines.append("")
    lines.extend(_formula_lines(summary.formula))
    lines.append("")

    if len(summary.p_table) > 0:
        p_col = summary.p_table.columns[-1]
        lines.append("Parametric coefficients:")
        lines.extend(_coefmat_lines(summary.p_table, p_col=p_col))
        lines.append("")

    if summary.s_table is not None and len(summary.s_table) > 0:
        stat_name = "F" if summary.scale_estimated else "Chi.sq"
        display = summary.s_table.set_index("label")[
            ["edf", "ref_df", "wald_stat", "p_value"]
        ]
        display.columns = ["edf", "Ref.df", stat_name, "p-value"]
        lines.append("Approximate significance of smooth terms:")
        lines.extend(_coefmat_lines(display, p_col="p-value"))
        lines.append("")

    if summary.rank is not None and summary.rank < summary.np:
        lines.append(f"Rank: {summary.rank}/{summary.np}")

    tail_bits = []
    if summary.r_sq is not None:
        tail_bits.append(f"R-sq.(adj) = {_fmt_g(summary.r_sq, 3)}")
    if summary.dev_expl is not None:
        tail_bits.append(f"Deviance explained = {summary.dev_expl * 100.0:.3g}%")
    if tail_bits:
        lines.append("   ".join(tail_bits))

    score_bits = []
    if summary.method is not None and summary.sp_criterion is not None:
        score_bits.append(f"{summary.method} = {_fmt_g(summary.sp_criterion)}")
    score_bits.append(f"Scale est. = {_fmt_g(summary.scale)}")
    score_bits.append(f"n = {summary.n}")
    lines.append("  ".join(score_bits))
    return lines


def build_summary_lines(model) -> list[str]:
    from ..inference.summary import summary_gam

    return summary_lines_from_gam_summary(summary_gam(model))


def summary_text(model_or_summary) -> str:
    from ..inference.summary import GAMSummary

    if isinstance(model_or_summary, GAMSummary):
        return "\n".join(summary_lines_from_gam_summary(model_or_summary))
    return "\n".join(build_summary_lines(model_or_summary))


def print_summary(model, *, dispersion=None, freq=False, re_test=True):
    from ..inference.summary import summary_gam

    summary = summary_gam(
        model, dispersion=dispersion, freq=freq, re_test=re_test
    )
    print(summary_text(summary))
    return summary


__all__ = [
    "build_summary_lines",
    "print_summary",
    "summary_lines_from_gam_summary",
    "summary_text",
]
