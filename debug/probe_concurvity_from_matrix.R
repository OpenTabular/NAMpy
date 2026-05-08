suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("usage: Rscript probe_concurvity_from_matrix.R X.csv coef.csv starts.csv stops.csv out.json")
}

X <- as.matrix(read.csv(args[[1]], header = FALSE))
coef <- as.numeric(read.csv(args[[2]], header = FALSE)[, 1])
starts <- as.integer(read.csv(args[[3]], header = FALSE)[, 1])
stops <- as.integer(read.csv(args[[4]], header = FALSE)[, 1])

X <- X[rowSums(is.na(X)) == 0, , drop = FALSE]
X <- qr.R(qr(X, tol = 0, LAPACK = FALSE))

m <- length(starts)
n.measures <- 3
measure.names <- c("worst", "observed", "estimate")
conc <- list()
for (i in 1:n.measures) conc[[i]] <- matrix(1, m, m)
for (i in 1:m) {
  Xi <- X[, starts[i]:stops[i], drop = FALSE]
  r <- ncol(Xi)
  for (j in 1:m) if (i != j) {
    Xj <- X[, starts[j]:stops[j], drop = FALSE]
    R <- qr.R(qr(cbind(Xi, Xj), LAPACK = FALSE, tol = 0))[, -(1:r), drop = FALSE]
    Rt <- qr.R(qr(R, tol = 0))
    conc[[1]][i, j] <- svd(forwardsolve(t(Rt), t(R[1:r, , drop = FALSE])))$d[1]^2
    beta <- coef[starts[j]:stops[j]]
    conc[[2]][i, j] <- sum((R[1:r, , drop = FALSE] %*% beta)^2) / sum((Rt %*% beta)^2)
    conc[[3]][i, j] <- sum(R[1:r, ]^2) / sum(R^2)
  }
}
names(conc) <- measure.names
write_json(
  list(
    worst = unname(conc$worst),
    observed = unname(conc$observed),
    estimate = unname(conc$estimate)
  ),
  args[[5]],
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE
)
