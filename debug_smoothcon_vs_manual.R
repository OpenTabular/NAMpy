suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]], stringsAsFactors = FALSE)
out <- args[[2]]

x0 <- d$x0; x1 <- d$x1

# Use smoothCon on the te object - get internal state
sm_te <- smoothCon(te(x0, x1, bs=c("cr","cr"), k=c(5,5)), d, absorb.cons=TRUE, scale.penalty=TRUE)[[1]]

# Now try smoothCon on each marginal separately (as smoothCon does internally)
sm0_sc <- smoothCon(s(x0, k=5, bs="cr"), d, absorb.cons=FALSE, scale.penalty=FALSE)[[1]]
sm1_sc <- smoothCon(s(x1, k=5, bs="cr"), d, absorb.cons=FALSE, scale.penalty=FALSE)[[1]]

# The XP is stored in sm_te$XP
cat("sm_te$XP is NULL:", is.null(sm_te$XP), "\n")
cat("length(sm_te$XP):", length(sm_te$XP), "\n")

# Get the XP matrices used by smoothCon's te
# sm_te$XP[[1]] is the np transform for marginal 1
if (!is.null(sm_te$XP) && length(sm_te$XP)>=1) {
  cat("XP[[1]][1,1]:", sm_te$XP[[1]][1,1], "\n")
  cat("XP[[2]][1,1]:", sm_te$XP[[2]][1,1], "\n")
}

# What do the te marginal bases look like in smoothCon?
cat("sm_te$margin[[1]]$X[1,1]:", sm_te$margin[[1]]$X[1,1], "\n")
cat("sm_te$margin[[2]]$X[1,1]:", sm_te$margin[[2]]$X[1,1], "\n")

# What about the S matrices of the marginals?
cat("sm_te$margin[[1]]$S[[1]][1,1]:", sm_te$margin[[1]]$S[[1]][1,1], "\n")

# Build marginal with smooth.construct
spec0 <- s(x0, k=5, bs="cr")
sm0 <- smooth.construct(spec0, list(x0=x0), list(x0=NULL))

cat("sm0$X[1,1] (raw smooth.construct):", sm0$X[1,1], "\n")
cat("sm0$S[[1]][1,1] (raw smooth.construct):", sm0$S[[1]][1,1], "\n")

# np reparam on sm0
np0 <- ncol(sm0$X)
knt0 <- seq(min(x0), max(x0), length=np0)
pd0 <- list(x0=knt0)
names(pd0) <- sm0$term
sv0 <- svd(Predict.matrix(sm0, pd0))
XP0 <- sv0$v %*% (t(sv0$u) / sv0$d)

cat("XP0 from sm0 (raw sc):", XP0[1,1], "\n")
cat("sm_te$XP[[1]][1,1]:", if(!is.null(sm_te$XP)) sm_te$XP[[1]][1,1] else NA, "\n")

# Compare XP
if (!is.null(sm_te$XP) && !is.null(sm_te$XP[[1]])) {
  cat("XP diff max:", max(abs(XP0 - sm_te$XP[[1]])), "\n")
}

# What about sm_te$margin[[1]]?
# After np reparam in smooth.construct.tensor.smooth.spec, the margin $X gets updated to X %*% XP
# So sm_te$margin[[1]]$X should be X0_raw %*% XP
X0_np_manual <- sm0$X %*% XP0
cat("X0_np_manual[1,1]:", X0_np_manual[1,1], "\n")
cat("sm_te$margin[[1]]$X[1,1]:", sm_te$margin[[1]]$X[1,1], "\n")
cat("Match:", max(abs(X0_np_manual - sm_te$margin[[1]]$X)) < 1e-12, "\n")

# S0_np from sm_te$margin[[1]]$S[[1]] vs manual
S0_np_manual <- t(XP0) %*% sm0$S[[1]] %*% XP0
cat("S0_np_manual[1,1]:", S0_np_manual[1,1], "\n")
cat("sm_te$margin[[1]]$S[[1]][1,1]:", sm_te$margin[[1]]$S[[1]][1,1], "\n")

# After eigenvalue norm:
ev0 <- eigen(S0_np_manual, symmetric=TRUE, only.values=TRUE)$values[1]
S0_norm_manual <- S0_np_manual / ev0
cat("S0_norm_manual ev0:", ev0, "\n")
cat("S0_norm after smoothCon margin ev:", eigen(sm_te$margin[[1]]$S[[1]], symmetric=TRUE, only.values=TRUE)$values[1], "\n")

# Tensor product before scale.penalty
X_te <- sm0$X %*% XP0
X_te2 <- smoothCon(s(x1, k=5, bs="cr"), d, absorb.cons=FALSE, scale.penalty=FALSE)[[1]]$X
# Actually use marginals directly
X_te_marginals <- tensor.prod.model.matrix(list(sm_te$margin[[1]]$X, sm_te$margin[[2]]$X))
cat("X_te from marginals[1,1]:", X_te_marginals[1,1], "\n")
cat("sm_te$X BEFORE constraint X[1,1] (after scale penalty, pre-constraint)?", "\n")
# we can reconstruct by calling smooth.construct.tensor directly
sm_te_raw <- smooth.construct(te(x0, x1, bs=c("cr","cr"), k=c(5,5)), d, knots=NULL)
cat("smooth.construct te X[1,1]:", sm_te_raw$X[1,1], "\n")
cat("smooth.construct te S[[1]][1,1]:", sm_te_raw$S[[1]][1,1], "\n")
cat("smooth.construct te S[[2]][1,1]:", sm_te_raw$S[[2]][1,1], "\n")

write_json(list(
  sc_S0 = sm_te$S[[1]][1,1],
  sc_S1 = sm_te$S[[2]][1,1],
  sc_X = sm_te$X[1,1],
  te_raw_X = sm_te_raw$X[1,1],
  te_raw_S0 = sm_te_raw$S[[1]][1,1],
  te_raw_S1 = sm_te_raw$S[[2]][1,1],
  marginal_S0 = sm_te$margin[[1]]$S[[1]][1,1]
), out, auto_unbox=TRUE, digits=17)
