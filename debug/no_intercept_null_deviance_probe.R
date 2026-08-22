# Probe: object$null.deviance / dev.expl for a no-intercept gaussian gam.
# Usage: Rscript debug/no_intercept_null_deviance_probe.R <csv_path>
# Companion to tests/parity/test_mgcv_summary_parity.py::gaussian_no_intercept_reml.
suppressPackageStartupMessages(library(mgcv))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]])
fit <- gam(y ~ x0 + s(x1, bs = "cr", k = 8) - 1, data = d, method = "REML")
cat("intercept attr:", attr(fit$pterms, "intercept"), "\n")
cat("null.deviance:", sprintf("%.10f", fit$null.deviance), "\n")
cat("deviance:", sprintf("%.10f", fit$deviance), "\n")
cat("dev.expl:", sprintf("%.10f", summary(fit)$dev.expl), "\n")
y <- d$y
cat("sum(y^2):", sprintf("%.10f", sum(y^2)), "\n")
cat("sum((y-mean(y))^2):", sprintf("%.10f", sum((y - mean(y))^2)), "\n")
