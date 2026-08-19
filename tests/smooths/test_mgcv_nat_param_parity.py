"""Focused nat.param(type=1) parity vs mgcv (promoted from
debug/nat_param_type1_probe.py per todo C3).

Invariant policy: simple-eigenvalue directions must match up to column sign;
inside repeated eigenspaces only the represented subspace (projector) is a
parity target, never the raw eigenvector orientation.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nampy.gam.splines.basis.natparam import nat_param_type1
from tests.mgcv_parity_utils import _make_fs_data_4levels
from tests.smooths.test_mgcv_raw_constructor_parity import _build_runtime_term

R_SCRIPT = shutil.which("Rscript")

pytestmark = [
    pytest.mark.surface_regression,
    pytest.mark.skipif(R_SCRIPT is None, reason="Rscript required for mgcv parity"),
]


def _read_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", ndmin=2)


def _align_column_signs(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    signs = np.sign(np.sum(observed * expected, axis=0))
    signs[signs == 0.0] = 1.0
    return observed * signs


def _projector(matrix: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(np.asarray(matrix, dtype=np.float64), mode="reduced")
    return Q @ Q.T


def _run_r(script: str, temp: Path) -> None:
    path = temp / "probe.R"
    path.write_text(script.strip() + "\n", encoding="utf-8")
    subprocess.run(
        [R_SCRIPT, str(path), str(temp)], check=True, capture_output=True, text=True
    )


def test_nat_param_type1_matches_mgcv_up_to_column_sign():
    """Simple penalized directions match by sign; the repeated null block by span."""
    rng = np.random.default_rng(732)
    X = rng.normal(size=(72, 5))
    root = rng.normal(size=(3, 5))
    S = root.T @ root

    actual = nat_param_type1(X, S, rank=3, unit_fnorm=True)

    with tempfile.TemporaryDirectory(prefix="nampy-nat-param-") as temp_name:
        temp = Path(temp_name)
        np.savetxt(temp / "X.csv", X, delimiter=",", fmt="%.17g")
        np.savetxt(temp / "S.csv", S, delimiter=",", fmt="%.17g")
        _run_r(
            """
args <- commandArgs(trailingOnly=TRUE)
d <- args[[1]]
library(mgcv)
X <- as.matrix(read.csv(file.path(d, "X.csv"), header=FALSE))
S <- as.matrix(read.csv(file.path(d, "S.csv"), header=FALSE))
rp <- mgcv:::nat.param(X, S, rank=3, type=1, unit.fnorm=TRUE)
write.table(rp$X, file.path(d, "Xn.csv"), row.names=FALSE, col.names=FALSE, sep=",")
write.table(rp$P, file.path(d, "P.csv"), row.names=FALSE, col.names=FALSE, sep=",")
write.table(matrix(rp$D, nrow=1), file.path(d, "D.csv"),
            row.names=FALSE, col.names=FALSE, sep=",")
""",
            temp,
        )
        expected_X = _read_matrix(temp / "Xn.csv")
        expected_P = _read_matrix(temp / "P.csv")
        expected_D = _read_matrix(temp / "D.csv").ravel()

    np.testing.assert_allclose(
        np.asarray(actual["D"], dtype=np.float64), expected_D, rtol=1e-10, atol=1e-12
    )
    rank = 3
    actual_X = np.asarray(actual["X"], np.float64)
    actual_P = np.asarray(actual["P"], np.float64)
    aligned_X = _align_column_signs(actual_X[:, :rank], expected_X[:, :rank])
    np.testing.assert_allclose(
        aligned_X, expected_X[:, :rank], rtol=1e-8, atol=1e-10
    )
    aligned_P = _align_column_signs(actual_P[:, :rank], expected_P[:, :rank])
    np.testing.assert_allclose(
        aligned_P, expected_P[:, :rank], rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        _projector(actual_X[:, rank:]),
        _projector(expected_X[:, rank:]),
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        _projector(actual_P[:, rank:]),
        _projector(expected_P[:, rank:]),
        rtol=0.0,
        atol=1e-10,
    )


def test_nat_param_type1_fs_base_matches_mgcv_subspace_invariants():
    """
    Default tp factor-smooth base: the repeated null eigenspace may rotate
    with the LAPACK build, so compare represented subspaces (projectors) and
    the eigenvalue vector, never raw orientation.
    """
    data = _make_fs_data_4levels()
    term, _model_X, _ = _build_runtime_term(data, 'y ~ s(f, x, bs="fs", k=6)')
    B0, S0, _ = term._base_constructor_fit_matrices()
    actual = nat_param_type1(B0, S0, rank=4, unit_fnorm=True)

    with tempfile.TemporaryDirectory(prefix="nampy-fs-nat-param-") as temp_name:
        temp = Path(temp_name)
        np.savetxt(temp / "B0.csv", B0, delimiter=",", fmt="%.17g")
        np.savetxt(temp / "S0.csv", S0, delimiter=",", fmt="%.17g")
        _run_r(
            """
args <- commandArgs(trailingOnly=TRUE)
d <- args[[1]]
library(mgcv)
B0 <- as.matrix(read.csv(file.path(d, "B0.csv"), header=FALSE))
S0 <- as.matrix(read.csv(file.path(d, "S0.csv"), header=FALSE))
rp <- mgcv:::nat.param(B0, S0, rank=4, type=1, unit.fnorm=TRUE)
write.table(rp$X, file.path(d, "Xn.csv"), row.names=FALSE, col.names=FALSE, sep=",")
write.table(matrix(rp$D, nrow=1), file.path(d, "D.csv"),
            row.names=FALSE, col.names=FALSE, sep=",")
""",
            temp,
        )
        expected_X = _read_matrix(temp / "Xn.csv")
        expected_D = _read_matrix(temp / "D.csv").ravel()

    actual_X = np.asarray(actual["X"], dtype=np.float64)
    np.testing.assert_allclose(
        np.asarray(actual["D"], dtype=np.float64), expected_D, rtol=1e-8, atol=1e-10
    )
    rank = 4
    # Penalized range space and null space are parity targets as subspaces.
    np.testing.assert_allclose(
        _projector(actual_X[:, :rank]),
        _projector(expected_X[:, :rank]),
        rtol=0.0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        _projector(actual_X[:, rank:]),
        _projector(expected_X[:, rank:]),
        rtol=0.0,
        atol=1e-8,
    )
