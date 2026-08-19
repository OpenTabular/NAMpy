"""Is mgcv's nat.param(type=1) null-column order deterministic for ps bases?

For each case: build the ps base design (via NAMpy's constructor, which matches
R's raw constructor), run R's nat.param, and classify each null column as
constant-like or linear-like by correlation with [1, x]. Repeat with permuted
rows and mirrored x to see whether R's order is stable. Also print NAMpy's
order for the same inputs.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/ad32/projects/package/NAMpy")

import numpy as np
import pandas as pd

from nampy.gam.smooths.categorical.fs import _penalty_rank_from_base_term
from nampy.gam.splines.basis.natparam import nat_param_type1
from tests.smooths.test_mgcv_raw_constructor_parity import _build_runtime_term

R_CODE = """
library(mgcv)
d <- commandArgs(TRUE)[1]
B0 <- as.matrix(read.csv(file.path(d, "B0.csv"), header=FALSE))
S0 <- as.matrix(read.csv(file.path(d, "S0.csv"), header=FALSE))
rank <- as.integer(commandArgs(TRUE)[2])
rp <- mgcv:::nat.param(B0, S0, rank=rank, type=1, unit.fnorm=TRUE)
write.table(rp$X, file.path(d, "Xn.csv"), row.names=FALSE, col.names=FALSE, sep=",")
cat("ok")
"""


def classify(col, x):
    """Return 'const' or 'linear' by which regressor explains the column."""
    one = np.ones_like(x)
    xc = x - x.mean()
    c_const = abs(np.dot(col, one)) / (np.linalg.norm(col) * np.linalg.norm(one))
    c_lin = abs(np.dot(col, xc)) / (np.linalg.norm(col) * np.linalg.norm(xc))
    return "const" if c_const > c_lin else "linear", c_const, c_lin


def r_nat_param(B0, S0, rank):
    with tempfile.TemporaryDirectory() as tmpd:
        tp = Path(tmpd)
        np.savetxt(tp / "B0.csv", B0, delimiter=",", fmt="%.17g")
        np.savetxt(tp / "S0.csv", S0, delimiter=",", fmt="%.17g")
        rf = tp / "p.R"
        rf.write_text(R_CODE)
        res = subprocess.run(
            ["Rscript", str(rf), str(tp), str(rank)],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(res.stderr[-300:])
        return np.loadtxt(tp / "Xn.csv", delimiter=",", ndmin=2)


def base_matrices(seed, n, k):
    rng = np.random.default_rng(seed)
    f = rng.choice(np.array(["a", "b", "c"], dtype=object), size=n)
    x = rng.uniform(-1.5, 1.5, size=n)
    y = np.sin(x) + rng.normal(scale=0.08, size=n)
    data = pd.DataFrame({"y": y, "f": f, "x": x})
    term, _, _ = _build_runtime_term(
        data, f'y ~ s(f, x, bs="fs", xt=list(bs="ps", m=2, k={k}))'
    )
    B0, S0, _ = term._base_constructor_fit_matrices()
    B0 = np.asarray(B0, dtype=np.float64)
    S0 = np.asarray(S0, dtype=np.float64)
    rank = int(_penalty_rank_from_base_term(term._base_term, B0, S0))
    return B0, S0, rank, np.asarray(data["x"], dtype=np.float64)


for seed, n, k in [(1711, 180, 7), (1, 120, 7), (2, 150, 6), (3, 200, 8), (4, 90, 7)]:
    B0, S0, rank, x = base_matrices(seed, n, k)
    variants = {
        "orig": (B0, x),
        "rowperm": None,
        "mirror": None,
    }
    rng = np.random.default_rng(99)
    perm = rng.permutation(B0.shape[0])
    variants["rowperm"] = (B0[perm], x[perm])
    variants["mirror"] = (B0[::-1].copy(), x[::-1].copy())

    line = [f"seed={seed} n={n} k={k}:"]
    for name, (B, xv) in variants.items():
        Xn_r = r_nat_param(B, S0, rank)
        kinds = [classify(Xn_r[:, rank + j], xv)[0] for j in range(B.shape[1] - rank)]
        line.append(f"R-{name}=[{','.join(kinds)}]")
    rp = nat_param_type1(B0, S0, rank=rank, unit_fnorm=True)
    Xn_p = np.asarray(rp["X"], dtype=np.float64)
    kinds_p = [classify(Xn_p[:, rank + j], x)[0] for j in range(B0.shape[1] - rank)]
    line.append(f"nampy=[{','.join(kinds_p)}]")
    print("  ".join(line))
