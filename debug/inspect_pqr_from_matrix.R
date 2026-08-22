args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("usage: Rscript inspect_pqr_from_matrix.R <x_csv> <cov_csv> <scale> [data_csv formula prefit_x_bin fit_r_bin]")
}
suppressPackageStartupMessages(library(mgcv))
X <- as.matrix(read.csv(args[[1]], header = FALSE))
Vp <- as.matrix(read.csv(args[[2]], header = FALSE))
scale <- as.numeric(args[[3]])
qrx <- mgcv:::pqr(X, nt = 1)
R <- mgcv:::pqr.R(qrx)
R[, qrx$pivot] <- R
cat("R dim", dim(R), "\n")
cat("edf2 sum", sum(rowSums(Vp * crossprod(R)) / scale), "\n")
cat("pivot head", head(qrx$pivot, 20), "\n")
if (length(args) >= 5) {
  normalize_formula_text <- function(x) {
    x <- gsub("\\[", "c(", x)
    x <- gsub("\\]", ")", x)
    x <- gsub("\\bTrue\\b", "TRUE", x)
    x <- gsub("\\bFalse\\b", "FALSE", x)
    x <- gsub("\\bNone\\b", "NULL", x)
    x
  }
  data <- read.csv(args[[4]], stringsAsFactors = FALSE)
  for (nm in names(data)) {
    if (is.character(data[[nm]])) data[[nm]] <- factor(data[[nm]])
  }
  fit <- gam(as.formula(normalize_formula_text(args[[5]])), data = data, family = gaussian(), method = "REML")
  prefit <- gam(as.formula(normalize_formula_text(args[[5]])), data = data, family = gaussian(), method = "REML", fit = FALSE)
  cat("fit weights range", range(fit$weights), "\n")
  cat("prior weights range", range(fit$prior.weights), "\n")
  cat("input X vs prefit X max", max(abs(X - prefit$X)), "\n")
  qrp <- mgcv:::pqr(prefit$X, nt = 1)
  Rp <- mgcv:::pqr.R(qrp)
  Rp[, qrp$pivot] <- Rp
  cat("fit R vs pqr(prefit X) max", max(abs(fit$R - Rp)), "\n")
  cat("prefit pivot head", head(qrp$pivot, 20), "\n")
  cat("fit R edf2 public Vp", sum(rowSums(fit$Vp * crossprod(fit$R)) / fit$scale), "\n")
  cat("fit edf2 sum", sum(fit$edf2), "\n")
  cat("fit R vs pqr(X) max", max(abs(fit$R - R)), "\n")
  qrw <- mgcv:::pqr(sqrt(fit$weights) * X, nt = 1)
  Rw <- mgcv:::pqr.R(qrw)
  Rw[, qrw$pivot] <- Rw
  cat("weighted R edf2 public Vp", sum(rowSums(fit$Vp * crossprod(Rw)) / fit$scale), "\n")
  cat("fit R vs pqr(weighted X) max", max(abs(fit$R - Rw)), "\n")
  cat("weighted pivot head", head(qrw$pivot, 20), "\n")
  if (length(args) >= 7) {
    con <- file(args[[6]], "wb")
    writeBin(as.double(prefit$X), con, size = 8)
    close(con)
    con <- file(args[[7]], "wb")
    writeBin(as.double(fit$R), con, size = 8)
    close(con)
  }
}
