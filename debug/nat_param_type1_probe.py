"""Compare NAMpy's natural parameterization with base R/mgcv internals.

The arbitrary full-rank case checks simple eigendirections up to column sign.
The default thin-plate factor-smooth case records the accepted boundary:
vectors in its repeated null eigenspace may rotate with the BLAS/LAPACK build,
although the represented subspace and downstream behavior remain the parity
targets.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes

from nampy.splines.basis.natparam import (
    _r_linpack_qr,
    _r_triangular_solve,
    nat_param_type1,
)
from tests.mgcv_parity_utils import _make_fs_data_4levels
from tests.smooths.test_mgcv_raw_constructor_parity import _build_runtime_term


def _read_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", ndmin=2)


def _align_column_signs(
    observed: np.ndarray, expected: np.ndarray
) -> np.ndarray:
    signs = np.sign(np.sum(observed * expected, axis=0))
    signs[signs == 0.0] = 1.0
    return observed * signs


def _projector(matrix: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(matrix, mode="reduced")
    return Q @ Q.T


def main() -> None:
    rng = np.random.default_rng(732)
    X = rng.normal(size=(72, 5))
    root = rng.normal(size=(3, 5))
    S = root.T @ root
    tol = np.finfo(float).eps**0.8

    Q, R = _r_linpack_qr(X, tol)
    actual = nat_param_type1(X, S, rank=3, unit_fnorm=True)

    with tempfile.TemporaryDirectory(prefix="nampy-nat-param-") as temp_name:
        temp = Path(temp_name)
        np.savetxt(temp / "X.csv", X, delimiter=",", fmt="%.17g")
        np.savetxt(temp / "S.csv", S, delimiter=",", fmt="%.17g")
        script = temp / "probe.R"
        script.write_text(
            """
args <- commandArgs(trailingOnly=TRUE)
d <- args[[1]]
X <- as.matrix(read.csv(file.path(d, "X.csv"), header=FALSE))
S <- as.matrix(read.csv(file.path(d, "S.csv"), header=FALSE))
qrx <- qr(X, tol=.Machine$double.eps^.8)
rp <- mgcv:::nat.param(X, S, rank=3, type=1, unit.fnorm=TRUE)
write.table(qr.Q(qrx, complete=FALSE), file.path(d, "Q.csv"),
            row.names=FALSE, col.names=FALSE, sep=",")
write.table(qr.R(qrx), file.path(d, "R.csv"),
            row.names=FALSE, col.names=FALSE, sep=",")
write.table(rp$X, file.path(d, "Xn.csv"),
            row.names=FALSE, col.names=FALSE, sep=",")
write.table(rp$P, file.path(d, "P.csv"),
            row.names=FALSE, col.names=FALSE, sep=",")
write.table(matrix(rp$D, nrow=1), file.path(d, "D.csv"),
            row.names=FALSE, col.names=FALSE, sep=",")
""".strip()
            + "\n",
            encoding="utf-8",
        )
        subprocess.run(["Rscript", str(script), str(temp)], check=True)

        expected = {
            "Q": _read_matrix(temp / "Q.csv"),
            "R": _read_matrix(temp / "R.csv"),
            "X": _read_matrix(temp / "Xn.csv"),
            "P": _read_matrix(temp / "P.csv"),
            "D": _read_matrix(temp / "D.csv").ravel(),
        }

    for name, observed in (
        ("Q", Q),
        ("R", R),
        ("X", actual["X"]),
        ("P", actual["P"]),
        ("D", actual["D"]),
    ):
        observed = np.asarray(observed)
        difference = np.max(np.abs(observed - expected[name]))
        print(name, "raw_max_abs_difference", float(difference))
        if observed.ndim == 2 and observed.shape[1] == expected[name].shape[1]:
            aligned = _align_column_signs(observed, expected[name])
            aligned_difference = np.max(np.abs(aligned - expected[name]))
            print(
                name,
                "column_sign_aligned_max_abs_difference",
                float(aligned_difference),
            )
    print(
        "X infinity norm",
        float(np.linalg.norm(actual["X"], ord=np.inf)),
        float(np.linalg.norm(expected["X"], ord=np.inf)),
    )
    observed_norm = np.linalg.norm(actual["X"], axis=0)
    expected_norm = np.linalg.norm(expected["X"], axis=0)
    correlations = actual["X"].T @ expected["X"]
    correlations /= observed_norm[:, None] * expected_norm[None, :]
    print("absolute column correlations")
    print(np.round(np.abs(correlations), 12))

    data = _make_fs_data_4levels()
    term, model_X, _ = _build_runtime_term(data, 'y ~ s(f, x, bs="fs", k=6)')
    B0, S0, _ = term._base_constructor_fit_matrices()
    base_actual = nat_param_type1(B0, S0, rank=4, unit_fnorm=True)

    with tempfile.TemporaryDirectory(prefix="nampy-fs-tp-") as temp_name:
        temp = Path(temp_name)
        data.to_csv(temp / "data.csv", index=False)
        np.savetxt(temp / "B0.csv", B0, delimiter=",", fmt="%.17g")
        np.savetxt(temp / "S0.csv", S0, delimiter=",", fmt="%.17g")
        script = temp / "fs_probe.R"
        script.write_text(
            """
args <- commandArgs(trailingOnly=TRUE)
d <- args[[1]]
library(mgcv)
dat <- read.csv(file.path(d, "data.csv"))
dat$f <- factor(dat$f)
base <- smooth.construct(s(x, bs="tp", k=6), dat, knots=NULL)
fs <- smooth.construct(s(f, x, bs="fs", k=6), dat, knots=NULL)
B0 <- as.matrix(read.csv(file.path(d, "B0.csv"), header=FALSE))
S0 <- as.matrix(read.csv(file.path(d, "S0.csv"), header=FALSE))
qrx <- qr(B0, tol=.Machine$double.eps^.8)
R <- qr.R(qrx)
RSR <- forwardsolve(t(R), t(forwardsolve(t(R), t(S0))))
input.rp <- mgcv:::nat.param(B0, S0, rank=4, type=1, unit.fnorm=TRUE)
out <- list(base.X=base$X, base.S=base$S[[1]], fs.Xb=fs$Xb,
            fs.P=fs$P, input.Q=qr.Q(qrx, complete=FALSE), input.R=R,
            input.RSR=RSR, input.X=input.rp$X, input.P=input.rp$P)
for (name in names(out)) {
  write.table(out[[name]], file.path(d, paste0(name, ".csv")),
              row.names=FALSE, col.names=FALSE, sep=",")
}
""".strip()
            + "\n",
            encoding="utf-8",
        )
        subprocess.run(["Rscript", str(script), str(temp)], check=True)
        base_expected = {
            name: _read_matrix(temp / f"{name}.csv")
            for name in (
                "base.X",
                "base.S",
                "fs.Xb",
                "fs.P",
                "input.Q",
                "input.R",
                "input.RSR",
                "input.X",
                "input.P",
            )
        }

    input_Q, input_R = _r_linpack_qr(B0, np.finfo(float).eps**0.8)
    input_tmp = _r_triangular_solve(input_R.T, S0.T, lower=True)
    input_RSR = _r_triangular_solve(input_R.T, input_tmp.T, lower=True)
    print("factor-smooth tp base")
    print("base X max_abs_difference", float(np.max(np.abs(B0 - base_expected["base.X"]))))
    print("base S max_abs_difference", float(np.max(np.abs(S0 - base_expected["base.S"]))))
    print("input Q max_abs_difference", float(np.max(np.abs(input_Q - base_expected["input.Q"]))))
    print("input R max_abs_difference", float(np.max(np.abs(input_R - base_expected["input.R"]))))
    print(
        "input RSR max_abs_difference",
        float(np.max(np.abs(input_RSR - base_expected["input.RSR"]))),
    )
    print(
        "same-input nat X max_abs_difference",
        float(np.max(np.abs(base_actual["X"] - base_expected["input.X"]))),
    )
    print(
        "same-input nat infinity norm",
        float(np.linalg.norm(base_actual["X"], ord=np.inf)),
        float(np.linalg.norm(base_expected["input.X"], ord=np.inf)),
    )
    print(
        "same-input null projector max_abs_difference",
        float(
            np.max(
                np.abs(
                    _projector(base_actual["X"][:, 4:])
                    - _projector(base_expected["input.X"][:, 4:])
                )
            )
        ),
    )
    print(
        "constructor nat infinity norm",
        float(np.linalg.norm(base_actual["X"], ord=np.inf)),
        float(np.linalg.norm(base_expected["fs.Xb"], ord=np.inf)),
    )
    metric = data["x"].to_numpy(dtype=np.float64)
    metric -= metric.mean()
    null_actual = base_actual["X"][:, 4:]
    centered_norm = np.linalg.norm(
        null_actual - null_actual.mean(axis=0, keepdims=True), axis=0
    )
    target = (
        np.column_stack([metric, np.ones(metric.size)])
        if centered_norm[0] > centered_norm[1]
        else np.column_stack([np.ones(metric.size), metric])
    )
    rotation, _ = orthogonal_procrustes(null_actual, target)
    aligned = base_actual["X"].copy()
    aligned[:, 4:] = null_actual @ rotation
    print(
        "compatibility-aligned infinity norm",
        float(np.linalg.norm(aligned, ord=np.inf)),
    )
    del model_X


if __name__ == "__main__":
    main()
