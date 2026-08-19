"""Localize partial-zero-weight wide EDF2 parity without comparing QR gauges."""

# ruff: noqa: E402

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nampy.gam.linalg.qr import mgcv_pqr_r
from tests.mgcv_parity_utils import _run_mgcv_snapshot
from tests.parity.test_mgcv_structured_unconditional_inference_parity import (
    _WIDE_CASES,
    _fit_case,
)


def main() -> None:
    case = next(
        item
        for item in _WIDE_CASES
        if item.case_id == "wide_fixed_te_partial_zero_weights"
    )
    data = case.data_factory()
    gam = _fit_case(case, data)
    snapshot = _run_mgcv_snapshot(
        data,
        case.formula,
        case.family,
        case.method,
        weights_column=case.weights_column,
        allow_live_run=True,
    )
    permuted_data = data.iloc[
        np.random.default_rng(20260831).permutation(len(data))
    ].reset_index(drop=True)
    permuted_snapshot = _run_mgcv_snapshot(
        permuted_data,
        case.formula,
        case.family,
        case.method,
        weights_column=case.weights_column,
        allow_live_run=True,
    )

    fit_state = gam.gam_result_.fit_core_solution.fit_state
    fit_result = gam.gam_result_.fit_core_solution.fit_result
    X = np.asarray(fit_state.X, dtype=np.float64)
    weights = np.asarray(fit_state.fisher_weights, dtype=np.float64)
    WX = np.sqrt(weights)[:, None] * X

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        source_path = tmp / "wx.csv"
        result_path = tmp / "r.csv"
        pivot_path = tmp / "pivot.csv"
        data_path = tmp / "data.csv"
        fit_r_path = tmp / "fit_r.csv"
        fit_vp_path = tmp / "fit_vp.csv"
        fit_x_path = tmp / "fit_x.csv"
        fit_scale_path = tmp / "fit_scale.csv"
        setup_x_path = tmp / "setup_x.csv"
        setup_p_path = tmp / "setup_p.csv"
        pd.DataFrame(WX).to_csv(source_path, index=False, header=False)
        data.to_csv(data_path, index=False)
        r_code = (
            "suppressPackageStartupMessages(library(mgcv));"
            "x<-as.matrix(read.csv(commandArgs(TRUE)[1],header=FALSE));"
            "q<-mgcv:::pqr(x);r<-mgcv:::pqr.R(q);r[,q$pivot]<-r;"
            "write.table(r,commandArgs(TRUE)[2],sep=',',row.names=FALSE,"
            "col.names=FALSE);"
            "write.table(q$pivot,commandArgs(TRUE)[3],sep=',',row.names=FALSE,"
            "col.names=FALSE)"
        )
        subprocess.run(
            [
                "Rscript",
                "-e",
                r_code,
                str(source_path),
                str(result_path),
                str(pivot_path),
            ],
            check=True,
        )
        r_reference = np.loadtxt(result_path, delimiter=",")
        pivot_reference = np.loadtxt(pivot_path, delimiter=",").astype(int)

        fit_code = (
            "suppressPackageStartupMessages(library(mgcv));"
            "a<-commandArgs(TRUE);d<-read.csv(a[1]);"
            "f<-gsub('\\\\[','c(',a[2]);f<-gsub('\\\\]',')',f);"
            "ga<-list(formula=as.formula(f),data=d,weights=d$w,method='REML');"
            "g<-do.call(gam,c(ga,list(fit=FALSE)));b<-do.call(gam,ga);"
            "write.table(b$R,a[3],sep=',',row.names=FALSE,col.names=FALSE);"
            "write.table(b$Vp,a[4],sep=',',row.names=FALSE,col.names=FALSE);"
            "write.table(predict(b,type='lpmatrix'),a[5],sep=',',row.names=FALSE,"
            "col.names=FALSE);"
            "write.table(b$sig2,a[6],sep=',',row.names=FALSE,col.names=FALSE);"
            "write.table(g$X,a[7],sep=',',row.names=FALSE,col.names=FALSE);"
            "if(!is.null(g$P))write.table(g$P,a[8],sep=',',row.names=FALSE,"
            "col.names=FALSE)"
        )
        subprocess.run(
            [
                "Rscript",
                "-e",
                fit_code,
                str(data_path),
                case.formula,
                str(fit_r_path),
                str(fit_vp_path),
                str(fit_x_path),
                str(fit_scale_path),
                str(setup_x_path),
                str(setup_p_path),
            ],
            check=True,
        )
        fit_r_reference = np.loadtxt(fit_r_path, delimiter=",")
        fit_vp_reference = np.loadtxt(fit_vp_path, delimiter=",")
        fit_x_reference = np.loadtxt(fit_x_path, delimiter=",")
        fit_scale_reference = float(np.loadtxt(fit_scale_path, delimiter=","))
        setup_x_reference = np.loadtxt(setup_x_path, delimiter=",")
        setup_p_reference = (
            None
            if not setup_p_path.exists()
            else np.loadtxt(setup_p_path, delimiter=",")
        )

    r_python = mgcv_pqr_r(WX)
    Vb = np.asarray(fit_result.cov_bayes, dtype=np.float64)
    scale = float(fit_result.scale)

    def edf2_total(
        root: np.ndarray,
        covariance: np.ndarray = Vb,
        covariance_scale: float = scale,
    ) -> float:
        crossproduct = np.asarray(root.T @ root, dtype=np.float64)
        return float(np.sum(covariance * crossproduct) / covariance_scale)

    print("shape", WX.shape)
    print("zero_weight_rows", np.flatnonzero(weights == 0.0).tolist())
    print("r_max_abs_difference", float(np.max(np.abs(r_python - r_reference))))
    print("r_pivot", pivot_reference.tolist())
    print("python_pqr_edf2", edf2_total(r_python))
    print("r_pqr_edf2", edf2_total(r_reference))
    print("direct_crossproduct_edf2", float(np.sum(Vb * (WX.T @ WX)) / scale))
    print(
        "mgcv_fit_r_edf2",
        edf2_total(fit_r_reference, fit_vp_reference, fit_scale_reference),
    )
    fit_wx_reference = np.sqrt(weights)[:, None] * fit_x_reference
    fit_r_from_python = mgcv_pqr_r(fit_wx_reference)
    print(
        "mgcv_x_python_pqr_edf2",
        edf2_total(fit_r_from_python, fit_vp_reference, fit_scale_reference),
    )
    print(
        "mgcv_fit_r_vs_python_pqr_max_abs",
        float(np.max(np.abs(fit_r_reference - fit_r_from_python))),
    )
    setup_wx_reference = np.sqrt(weights)[:, None] * setup_x_reference
    setup_r_from_python = mgcv_pqr_r(setup_wx_reference)
    print(
        "mgcv_fit_r_vs_setup_python_pqr_max_abs",
        float(np.max(np.abs(fit_r_reference - setup_r_from_python))),
    )
    print("python_x_vs_mgcv_setup_x_max_abs", float(np.max(np.abs(X - setup_x_reference))))
    print(
        "python_x_vs_mgcv_setup_x_gram_max_abs",
        float(np.max(np.abs(X @ X.T - setup_x_reference @ setup_x_reference.T))),
    )
    print("mgcv_setup_has_p", setup_p_reference is not None)
    print("mgcv_edf", float(snapshot["fit"]["edf_total"]))
    print("mgcv_edf2", float(np.sum(snapshot["fit"]["edf2"])))
    print("permuted_mgcv_edf", float(permuted_snapshot["fit"]["edf_total"]))
    print("permuted_mgcv_edf2", float(np.sum(permuted_snapshot["fit"]["edf2"])))
    print(
        "mgcv_anova",
        snapshot["parity"]["diagnostics"]["anova_smooth"]["values"],
    )
    print(
        "permuted_mgcv_anova",
        permuted_snapshot["parity"]["diagnostics"]["anova_smooth"]["values"],
    )


if __name__ == "__main__":
    main()
