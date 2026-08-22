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

from .summary_format import summary_lines_from_gam_summary


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
