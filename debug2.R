suppressPackageStartupMessages(library(mgcv))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]], stringsAsFactors = FALSE)

x0 <- d$x0; x1 <- d$x1

# Build marginals with smooth.construct
spec0 <- s(x0, k=5, bs="cr")
sm0 <- smooth.construct(spec0, list(x0=x0), list(x0=NULL))

cat("sm0$X[1,1]:", sm0$X[1,1], "\n")
cat("sm0$S[[1]][1,1]:", sm0$S[[1]][1,1], "\n")
cat("sm0$dim:", sm0$dim, "\n")
cat("sm0$noterp:", if(!is.null(sm0$noterp)) sm0$noterp else "NULL", "\n")
cat("sm0$term:", sm0$term, "\n")
cat("class(sm0):", class(sm0), "\n")

np0 <- ncol(sm0$X)
knt0 <- seq(min(x0), max(x0), length=np0)
pd0 <- list(); pd0[[sm0$term]] <- knt0
cat("knt0:", knt0[1], knt0[2], "\n")

Pred0 <- Predict.matrix(sm0, pd0)
cat("Pred0 shape:", nrow(Pred0), ncol(Pred0), "\n")
cat("Pred0[1,1]:", Pred0[1,1], "\n")
cat("Pred0[1,2]:", Pred0[1,2], "\n")
sv0 <- svd(Pred0)
cat("sv0$d:", sv0$d, "\n")
cat("sv0$u[1,1]:", sv0$u[1,1], "\n")
cat("sv0$v[1,1]:", sv0$v[1,1], "\n")
XP0 <- sv0$v %*% (t(sv0$u) / sv0$d)
cat("XP0[1,1]:", XP0[1,1], "\n")
cat("XP0[1,2]:", XP0[1,2], "\n")

# Now check what smooth.construct.tensor.smooth.spec does
# The te call gives us the np transform in XP
sm_te_raw <- smooth.construct(te(x0, x1, bs=c("cr","cr"), k=c(5,5)), d, knots=NULL)
cat("sm_te_raw$XP length:", length(sm_te_raw$XP), "\n")
if (length(sm_te_raw$XP) >= 1 && !is.null(sm_te_raw$XP[[1]])) {
  cat("sm_te_raw$XP[[1]][1,1]:", sm_te_raw$XP[[1]][1,1], "\n")
  cat("sm_te_raw$XP[[1]][1,2]:", sm_te_raw$XP[[1]][1,2], "\n")
}
cat("sm_te_raw$X[1,1]:", sm_te_raw$X[1,1], "\n")
cat("sm_te_raw$S[[1]][1,1]:", sm_te_raw$S[[1]][1,1], "\n")
cat("sm_te_raw$margin[[1]]$X[1,1]:", sm_te_raw$margin[[1]]$X[1,1], "\n")
cat("sm_te_raw$margin[[1]]$S[[1]][1,1]:", sm_te_raw$margin[[1]]$S[[1]][1,1], "\n")
cat("sm_te_raw$margin[[1]]$np:", if(!is.null(sm_te_raw$margin[[1]]$np)) sm_te_raw$margin[[1]]$np else "NULL", "\n")
