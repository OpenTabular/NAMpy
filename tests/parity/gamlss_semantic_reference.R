#!/usr/bin/env Rscript

# Generate small, deterministic cross-package semantic references for NAMpy's
# public GAMLSS parameters. mgcv remains the numerical smooth-fit authority;
# gamlss checks that the exposed (mu, sigma) columns have conventional GAMLSS
# distribution semantics.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("usage: gamlss_semantic_reference.R OUTPUT_DIRECTORY")
}

extra_lib <- Sys.getenv("GAMLSS_LIB_PATH", unset = "")
if (nzchar(extra_lib)) {
  .libPaths(c(extra_lib, .libPaths()))
}

suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(gamlss.dist))
options(digits = 17)

output_dir <- args[[1L]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

n <- 80L
x <- seq(-1.5, 1.5, length.out = n)
phase <- seq_len(n)
z <- sin(phase * 1.7) + 0.5 * cos(phase * 0.41)
reference_rows <- c(1L, 20L, 40L, 60L, 80L)

normal_data <- data.frame(
  x = x,
  y = 0.3 + 0.9 * x + exp(-0.4 + 0.2 * x) * z
)
normal_mgcv <- mgcv::gam(
  list(y ~ x, ~ x),
  family = mgcv::gaulss(),
  data = normal_data,
  method = "ML"
)
normal_gamlss <- gamlss::gamlss(
  y ~ x,
  sigma.formula = ~ x,
  family = gamlss.dist::NO(mu.link = "identity", sigma.link = "log"),
  data = normal_data,
  control = gamlss::gamlss.control(n.cyc = 200L, trace = FALSE)
)
normal_eta <- predict(normal_mgcv, type = "link")
normal_mgcv_mu <- normal_mgcv$family$linfo[[1L]]$linkinv(normal_eta[, 1L])
normal_mgcv_tau <- normal_mgcv$family$linfo[[2L]]$linkinv(normal_eta[, 2L])
normal_reference <- transform(
  normal_data,
  mgcv_mu = normal_mgcv_mu,
  mgcv_sigma = 1 / normal_mgcv_tau,
  gamlss_mu = predict(normal_gamlss, what = "mu", type = "response"),
  gamlss_sigma = predict(normal_gamlss, what = "sigma", type = "response")
)
normal_reference$gamlss_logpdf <- gamlss.dist::dNO(
  normal_reference$y,
  mu = normal_reference$gamlss_mu,
  sigma = normal_reference$gamlss_sigma,
  log = TRUE
)
normal_reference <- cbind(row = seq_len(n), normal_reference)[reference_rows, ]
write.csv(
  normal_reference,
  file.path(output_dir, "normal.csv"),
  row.names = FALSE
)

gamma_data <- data.frame(
  x = x,
  y = exp(0.2 + 0.5 * x) * exp(0.3 * z)
)
gamma_mgcv <- mgcv::gam(
  list(y ~ x, ~ x),
  family = mgcv::gammals(),
  data = gamma_data,
  method = "ML"
)
gamma_gamlss <- gamlss::gamlss(
  y ~ x,
  sigma.formula = ~ x,
  family = gamlss.dist::GA(mu.link = "log", sigma.link = "log"),
  data = gamma_data,
  control = gamlss::gamlss.control(n.cyc = 200L, trace = FALSE)
)
gamma_eta <- predict(gamma_mgcv, type = "link")
gamma_log_dispersion <- gamma_mgcv$family$linfo[[2L]]$linkinv(
  gamma_eta[, 2L]
)
gamma_reference <- transform(
  gamma_data,
  mgcv_mu = exp(gamma_eta[, 1L]),
  mgcv_sigma = sqrt(exp(gamma_log_dispersion)),
  gamlss_mu = predict(gamma_gamlss, what = "mu", type = "response"),
  gamlss_sigma = predict(gamma_gamlss, what = "sigma", type = "response")
)
gamma_reference$gamlss_logpdf <- gamlss.dist::dGA(
  gamma_reference$y,
  mu = gamma_reference$gamlss_mu,
  sigma = gamma_reference$gamlss_sigma,
  log = TRUE
)
gamma_reference <- cbind(row = seq_len(n), gamma_reference)[reference_rows, ]
write.csv(
  gamma_reference,
  file.path(output_dir, "gamma.csv"),
  row.names = FALSE
)

versions <- data.frame(
  package = c("mgcv", "gamlss", "gamlss.dist"),
  version = c(
    as.character(packageVersion("mgcv")),
    as.character(packageVersion("gamlss")),
    as.character(packageVersion("gamlss.dist"))
  )
)
write.csv(
  versions,
  file.path(output_dir, "versions.csv"),
  row.names = FALSE,
  quote = TRUE
)
