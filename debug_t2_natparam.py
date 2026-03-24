"""Debug t2 nat.param comparison between Python and R."""
import numpy as np
import subprocess
import tempfile
import json
from pathlib import Path
import sys
sys.path.insert(0, 'tests')

import pandas as pd
from nampy.gam.basis.tensor import t2_marginal_reparameterization
from nampy.gam.smooths.univariate.cubic_regression import SplineTerm1D

rng = np.random.default_rng(31)
data = pd.DataFrame({"x": rng.uniform(-2.0, 2.0, size=120)})

term = SplineTerm1D(feature="x", k=5, basis="cr")
term.fit(data[["x"]].to_numpy(dtype=np.float64), ["x"])

actual = t2_marginal_reparameterization(
    term._spline.raw_basis,
    term._spline.raw_penalty,
    knots=term._spline.knots,
)

got_X = np.column_stack([actual["B_range"], actual["B_null"]])
got_P = np.column_stack([actual["T_range"], actual["T_null"]])
print("Python X[0,:] =", got_X[0,:])
print("Python P[0,:] =", got_P[0,:])
print("Python rank =", actual["rank"])

# Run R
r_code = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
d <- read.csv(args[1])
out <- args[2]
k <- 5
sm <- smoothCon(s(x, bs="cr", k=k), d, absorb.cons=FALSE)[[1]]
np_res <- mgcv:::nat.param(sm$X, sm$S[[1]], rank=sm$rank, type=3, unit.fnorm=TRUE)
# Also do intermediate steps manually
er <- eigen(sm$S[[1]], symmetric=TRUE)
rank <- sm$rank
p <- ncol(sm$X)
E <- rep(1, p)
E[1:rank] <- sqrt(er$value[1:rank])
X_mid <- sm$X %*% er$vectors
col_norm <- colSums(X_mid^2) / E^2
av_norm <- mean(col_norm[1:rank])
for (i in (rank+1):p) E[i] <- sqrt(col_norm[i]/av_norm)
P_mid <- t(t(er$vectors)/E)
X_mid <- t(t(X_mid)/E)
# type==3 rotation
ind <- (rank+1):p
rind <- p:(rank+1)
Xn <- X_mid[,ind,drop=FALSE]
n <- nrow(Xn)
one <- rep(1,n)
Xn <- Xn - one%*%(t(one)%*%Xn)/n
um <- eigen(t(Xn)%*%Xn, symmetric=TRUE)
X_rot <- X_mid
P_rot <- P_mid
X_rot[,rind] <- X_mid[,ind,drop=FALSE]%*%um$vectors
P_rot[,rind] <- P_mid[,ind,drop=FALSE]%*%um$vectors
cat("X_rot[1,:] =", X_rot[1,], "\\n")
cat("P_rot[1,:] =", P_rot[1,], "\\n")
# unit.fnorm
ind2 <- 1:rank
scale <- 1/sqrt(mean(X_rot[,ind2]^2))
X_rot[,ind2] <- X_rot[,ind2]*scale
P_rot[ind2,] <- P_rot[ind2,]*scale
ind3 <- (rank+1):p
scalef <- 1/sqrt(mean(X_rot[,ind3]^2))
X_rot[,ind3] <- X_rot[,ind3]*scalef
P_rot[ind3,] <- P_rot[ind3,]*scalef
cat("After unit.fnorm X_rot[1,:] =", X_rot[1,], "\\n")
cat("nat.param X[1,:] =", np_res$X[1,], "\\n")
cat("max diff X:", max(abs(X_rot - np_res$X)), "\\n")
cat("max diff P:", max(abs(P_rot - np_res$P)), "\\n")
write_json(list(
  X = unname(np_res$X),
  P = unname(np_res$P),
  rank = np_res$rank,
  er_values = er$value,
  av_norm = av_norm
), out, auto_unbox=TRUE, digits=17)
"""

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir_path = Path(tmpdir)
    csv_path = tmpdir_path / "data.csv"
    json_path = tmpdir_path / "result.json"
    script_path = tmpdir_path / "debug.R"
    data.to_csv(csv_path, index=False)
    script_path.write_text(r_code)
    result = subprocess.run(['Rscript', str(script_path), str(csv_path), str(json_path)], capture_output=True, text=True)
    print("R stdout:", result.stdout)
    if result.returncode != 0:
        print("R error:", result.stderr[:500])
    else:
        r_vals = json.loads(json_path.read_text())
        want_X = np.asarray(r_vals["X"], dtype=np.float64)
        want_P = np.asarray(r_vals["P"], dtype=np.float64)
        print("R X[0,:] =", want_X[0,:])
        print("R P[0,:] =", want_P[0,:])
        print("max diff X:", np.max(np.abs(got_X - want_X)))
        print("max diff P:", np.max(np.abs(got_P - want_P)))
        print("X col 3 diff:", got_X[:,3] - want_X[:,3])
        print("P col 3 diff:", got_P[:,3] - want_P[:,3])
