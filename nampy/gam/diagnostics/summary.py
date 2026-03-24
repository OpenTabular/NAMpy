"""
Post-fit model summary generation.

:func:`build_summary_lines` produces a human-readable tabular summary of a
fitted GAM: family, link, effective degrees of freedom by term, smoothing
parameters, and goodness-of-fit statistics.
"""

from __future__ import annotations


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
    if not getattr(model, "_fitted", False):
        raise RuntimeError("Model is not fitted.")

    lines = []
    lines.append("General Smooth Model Summary")
    lines.append("=" * 65)
    lines.append(f"Family : {model.family.name}")
    lines.append(f"Link : {model.family.link_name}")
    lines.append(f"Smoothing method : {model._optim_method}")
    lines.append(f"n samples : {model.n_samples_}")
    if model.smoothing_score_ is not None:
        lines.append(f"Smoothing score : {model.smoothing_score_:.6g}")
    if model.offset_train_ is not None:
        lines.append("Offset : yes")
    lines.append("")

    lines.append(f"{'term':<20s} {'edf':>8s} {'k':>5s} {'sp':>20s}")
    lines.append("-" * 65)

    for i, tb in enumerate(model.term_blocks_):
        k_i = tb.coef_slice.stop - tb.coef_slice.start
        sp_vals = [model.smoothing_params[j] for j in tb.smoothing_indices]
        sp_txt = _format_sp_values(sp_vals)
        prefix = _term_prefix(tb)

        lines.append(
            f"{prefix}({tb.label:<16s}) "
            f"{model.edf_by_term_[i]:8.3f} "
            f"{k_i:5d} "
            f"{sp_txt:>20s}"
        )

    lines.append("-" * 65)
    if model.fit_intercept:
        lines.append(f"Intercept : {model.intercept_:.6g}")
    lines.append(f"EDF (total) : {model.edf_:.3f}")
    lines.append(f"Scale : {model.scale_:.6g}")
    if model.family.name == "gaussian":
        lines.append(f"RSS : {model.rss_:.6g}")
    else:
        lines.append(f"Deviance : {model.deviance_:.6g}")

    return lines


def summary_text(model) -> str:
    return "\n".join(build_summary_lines(model))


def print_summary(model) -> str:
    text = summary_text(model)
    print(text)
    return text


__all__ = ["build_summary_lines", "summary_text", "print_summary"]
