"""Plot NAMpy against the committed gaussian cubic-regression snapshot.

The mgcv numbers come from the ``gaussian_cr_uni_reml`` fixture. This script
refits that case in Python and does not call R.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tests.mgcv_parity_utils import (
    _fit_nampy_model,
    _run_mgcv_snapshot,
    get_parity_case,
    make_parity_case_data,
)

CASE_ID = "gaussian_cr_uni_reml"
STATIC = ROOT / "docs/_static/mgcv_parity_gaussian_cr.png"
TABLE = ROOT / "docs/_generated/mgcv_parity_gaussian_cr.inc"


def gaussian_cr_comparison():
    """Return aligned NAMpy and mgcv quantities for the committed snapshot."""
    case = get_parity_case(CASE_ID)
    data = make_parity_case_data(case.case_id)
    expected = _run_mgcv_snapshot(data, case.formula, case.family, case.method)
    model = _fit_nampy_model(data, case.formula, case.family, case.method)
    with contextlib.redirect_stdout(io.StringIO()):
        summary = model.summary()
    labels = [str(label) for label in summary.s_table["label"]]
    edf_nampy = np.asarray(summary.s_table["edf"], dtype=float)
    edf_mgcv = np.asarray(expected["fit"]["edf_by_term"], dtype=float)
    terms_nampy = np.asarray(model.predict(data, type="terms"), dtype=float)
    terms_mgcv = np.asarray(expected["predictions"]["terms"], dtype=float)
    response_nampy = np.asarray(model.predict(data, type="response"), dtype=float).reshape(-1)
    response_mgcv = np.asarray(expected["predictions"]["response"], dtype=float).reshape(-1)
    return {
        "formula": case.formula,
        "n": int(len(data)),
        "x0": np.asarray(data["x0"], dtype=float),
        "x1": np.asarray(data["x1"], dtype=float),
        "labels": labels,
        "terms_nampy": terms_nampy,
        "terms_mgcv": terms_mgcv,
        "response_nampy": response_nampy,
        "response_mgcv": response_mgcv,
        "edf_nampy": edf_nampy,
        "edf_mgcv": edf_mgcv,
        "edf_total_nampy": float(summary.edf_total),
        "edf_total_mgcv": float(expected["fit"]["edf_total"]),
        "reml_nampy": float(summary.sp_criterion),
        "reml_mgcv": float(expected["fit"]["criterion_value"]),
    }


def _table(comparison) -> str:
    def row(name, mgcv, nampy):
        return (
            f"   {name},{mgcv:.8g},{nampy:.8g},{abs(nampy - mgcv):.3g}"
        )

    lines = [
        ".. csv-table:: Gaussian cubic-regression snapshot",
        "   :header-rows: 1",
        "",
        "   Quantity,mgcv snapshot,NAMpy refit,Absolute difference",
    ]
    for label, mgcv, nampy in zip(
        comparison["labels"], comparison["edf_mgcv"], comparison["edf_nampy"]
    ):
        lines.append(row(f"EDF {label}", float(mgcv), float(nampy)))
    lines.append(
        row(
            "Total EDF",
            comparison["edf_total_mgcv"],
            comparison["edf_total_nampy"],
        )
    )
    lines.append(row("REML", comparison["reml_mgcv"], comparison["reml_nampy"]))
    response_gap = float(
        np.max(np.abs(comparison["response_nampy"] - comparison["response_mgcv"]))
    )
    term_gap = float(
        np.max(np.abs(comparison["terms_nampy"] - comparison["terms_mgcv"]))
    )
    lines.append(f"   Max absolute response difference,,,{response_gap:.3g}")
    lines.append(f"   Max absolute smooth-term difference,,,{term_gap:.3g}")
    return "\n".join(lines) + "\n"


def save_assets(comparison=None) -> None:
    """Write the comparison figure and the table included by the landing page."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if comparison is None:
        with contextlib.redirect_stdout(io.StringIO()):
            comparison = gaussian_cr_comparison()
    labels = comparison["labels"]
    covariates = (comparison["x0"], comparison["x1"])
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), constrained_layout=True)
    for axis, covariate, column, label in zip(
        axes[0], covariates, (0, 1), labels
    ):
        order = np.argsort(covariate)
        axis.plot(
            covariate[order],
            comparison["terms_mgcv"][order, column],
            color="#D55E00",
            linewidth=2.2,
            label="mgcv snapshot",
        )
        axis.plot(
            covariate[order],
            comparison["terms_nampy"][order, column],
            color="#0072B2",
            linewidth=1.3,
            linestyle="--",
            label="NAMpy",
        )
        axis.set(xlabel=label[2:-1], ylabel="Centered contribution", title=label)
        axis.legend(frameon=False)
    difference = comparison["response_nampy"] - comparison["response_mgcv"]
    axes[1, 0].plot(difference, color="#009E73", linewidth=1.0)
    axes[1, 0].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1, 0].set(
        xlabel="Training row",
        ylabel="NAMpy − mgcv",
        title="Response prediction difference",
    )
    positions = np.arange(len(labels))
    width = 0.36
    axes[1, 1].bar(
        positions - width / 2,
        comparison["edf_mgcv"],
        width=width,
        color="#D55E00",
        label="mgcv snapshot",
    )
    axes[1, 1].bar(
        positions + width / 2,
        comparison["edf_nampy"],
        width=width,
        color="#0072B2",
        label="NAMpy",
    )
    axes[1, 1].set_xticks(positions, labels)
    axes[1, 1].set(ylabel="Effective degrees of freedom", title="Smooth complexity")
    axes[1, 1].legend(frameon=False)
    figure.savefig(STATIC, dpi=150, facecolor="white")
    plt.close(figure)
    TABLE.write_text(_table(comparison))


if __name__ == "__main__":
    save_assets(gaussian_cr_comparison())
