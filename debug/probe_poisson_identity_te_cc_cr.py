from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nampy.gam.fit.linalg.stacked_qr as stacked_qr  # noqa: E402
from tests.gam_cartesian_matrix import MatrixCase, fit_model, make_data  # noqa: E402
from tests.mgcv_parity_utils import (  # noqa: E402
    R_SCRIPT,
    _normalize_python_formula_text,
    _run_mgcv_snapshot,
)

FORMULA = 'y ~ te(x0, x1, bs=["cc","cr"], k=[8,6], sp=[1.0,1.2])'
FAMILY = {"name": "poisson", "link": "identity"}


def _run_mgcv_trace(data, formula: str) -> str:
    formula_r = _normalize_python_formula_text(formula)
    code = r"""
args <- commandArgs(trailingOnly = TRUE)
csv_path <- args[[1]]
formula_text <- args[[2]]
suppressPackageStartupMessages(library(mgcv))
data <- read.csv(csv_path, stringsAsFactors = FALSE)
for (nm in names(data)) {
  if (is.character(data[[nm]])) data[[nm]] <- factor(data[[nm]])
}
fit <- gam(
  as.formula(formula_text),
  data = data,
  family = poisson(link = "identity"),
  method = "REML",
  control = gam.control(trace = TRUE)
)
cat("final_deviance", format(fit$deviance, digits = 17), "\n")
cat("final_loglik", format(as.numeric(logLik(fit)), digits = 17), "\n")
cat("coef", jsonlite::toJSON(unname(coef(fit)), auto_unbox = TRUE, digits = 17), "\n")
cat("fitted", jsonlite::toJSON(unname(fitted(fit)), auto_unbox = TRUE, digits = 17), "\n")
"""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "data.csv"
        data.to_csv(csv_path, index=False)
        proc = subprocess.run(
            [R_SCRIPT, "-e", code, str(csv_path), formula_r],
            check=False,
            text=True,
            capture_output=True,
        )
    return f"returncode={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"


def main() -> None:
    data = make_data("count")
    orig_signed = stacked_qr._signed_weight_rank_correction
    signed_rows = []

    def wrapped_signed(q1_negative_rows, *, rank_tol):
        _, sing_vals, _ = np.linalg.svd(
            np.pad(
                q1_negative_rows,
                ((0, max(0, q1_negative_rows.shape[1] + 1 - q1_negative_rows.shape[0])), (0, 0)),
            ),
            full_matrices=False,
        )
        delta = 1.0 - 2.0 * sing_vals * sing_vals
        signed_rows.append(
            {
                "n_neg": int(q1_negative_rows.shape[0]),
                "rank": int(q1_negative_rows.shape[1]),
                "min_delta": float(np.min(delta)) if delta.size else None,
                "rank_tol": float(rank_tol),
            }
        )
        return orig_signed(q1_negative_rows, rank_tol=rank_tol)

    stacked_qr._signed_weight_rank_correction = wrapped_signed
    case = MatrixCase(
        case_id="debug_poisson_identity_te_cc_cr",
        formula=FORMULA,
        family=FAMILY,
        method="fixed",
        data_kind="count",
    )
    try:
        gam = fit_model(case, data)
    finally:
        stacked_qr._signed_weight_rank_correction = orig_signed
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(
        data,
        case.formula,
        case.family,
        case.method,
        allow_live_run=True,
    )
    actual_fit = actual["fit"]
    expected_fit = expected["fit"]
    beta = np.asarray(actual_fit["coef_full"], dtype=np.float64)
    beta_ref = np.asarray(expected_fit["coef_full"], dtype=np.float64)
    fitted = np.asarray(actual["predictions"]["response"], dtype=np.float64)
    fitted_ref = np.asarray(expected["predictions"]["response"], dtype=np.float64)

    print("formula", FORMULA)
    for key in ("loglik", "deviance", "edf_total", "penalty_quadratic"):
        print(key, actual_fit.get(key), expected_fit.get(key))
    actual_diag = actual.get("parity", {}).get("diagnostics", {})
    expected_diag = expected.get("parity", {}).get("diagnostics", {})
    print("smooth_edf1", actual_diag.get("smooth_edf1"), expected_diag.get("smooth_edf1"))
    print("anova_smooth", actual_diag.get("anova_smooth"), expected_diag.get("anova_smooth"))
    print("coef max abs", float(np.max(np.abs(beta - beta_ref))))
    print("fitted max abs", float(np.max(np.abs(fitted - fitted_ref))))
    print("fitted min", float(np.min(fitted)), float(np.min(fitted_ref)))
    fit_result = gam.fit_core_solution_.fit_result
    print(
        "fit flags",
        {
            "converged": fit_result.converged,
            "failure_reason": getattr(fit_result, "failure_reason", None),
            "warnings": getattr(fit_result, "warnings", None),
        },
    )
    print("actual inner trace")
    trace = list(gam.fit_core_solution_.fit_result.inner_trace or [])
    print("trace_len", len(trace))
    print(json.dumps(trace[:12], indent=2))
    print("...")
    print(json.dumps(trace[-12:], indent=2))
    print("signed correction last rows")
    print(json.dumps(signed_rows[-20:], indent=2))
    print("mgcv trace")
    print(_run_mgcv_trace(data, FORMULA))


if __name__ == "__main__":
    main()
