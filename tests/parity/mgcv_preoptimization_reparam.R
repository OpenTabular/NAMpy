# Usage:
#   Rscript mgcv_preoptimization_reparam.R <csv_path> <output_json> <formula> <family> <method> <select>
#
# Fit an ordinary-family GAM in mgcv, reconstruct the exact estimate.gam
# setup objects at the fitted smoothing parameters, and serialize the
# corresponding gam.reparam output from mgcv/R/gam.fit3.r.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop(
    "Usage: Rscript mgcv_preoptimization_reparam.R <csv_path> <output_json> <formula> <family> <method> <select>"
  )
}

normalize_formula_text <- function(x) {
  x <- gsub("\\[", "c(", x)
  x <- gsub("\\]", ")", x)
  x <- gsub("\\bTrue\\b", "TRUE", x)
  x <- gsub("\\bFalse\\b", "FALSE", x)
  x <- gsub("\\bNone\\b", "NULL", x)
  x
}

serialize_numeric <- function(x) {
  if (is.null(x)) {
    NULL
  } else {
    unname(as.numeric(x))
  }
}

csv_path <- args[[1]]
output_json <- args[[2]]
formula_text <- normalize_formula_text(args[[3]])
family_name <- tolower(args[[4]])
method_name <- toupper(args[[5]])
select_flag <- tolower(args[[6]]) == "true"

mgcv_lib <- Sys.getenv("MGCV_LIB_PATH", "")
if (nzchar(mgcv_lib)) {
  .libPaths(c(mgcv_lib, .libPaths()))
}

suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))

data <- read.csv(csv_path, stringsAsFactors = FALSE)
for (nm in names(data)) {
  if (is.character(data[[nm]])) data[[nm]] <- factor(data[[nm]])
}

family_parts <- strsplit(family_name, ":", fixed = TRUE)[[1]]
family_key <- family_parts[[1]]
family_param <- if (length(family_parts) >= 2) family_parts[[2]] else NULL

family_obj <- switch(
  family_key,
  gaussian = gaussian(),
  binomial = binomial(link = if (is.null(family_param)) "logit" else family_param),
  poisson = poisson(link = if (is.null(family_param)) "log" else family_param),
  gamma = Gamma(link = if (is.null(family_param)) "log" else family_param),
  negbin_est = {
    theta <- if (is.null(family_param)) 1.0 else as.numeric(family_param)
    mgcv::nb(theta = -abs(theta), link = "log")
  },
  negbin = {
    theta <- if (is.null(family_param)) 1.0 else as.numeric(family_param)
    mgcv::nb(theta = theta, link = "log")
  },
  stop(sprintf("Unsupported family token for preoptimization reparameterization parity: %s", family_name))
)

fit_method <- if (tolower(method_name) == "fixed") "REML" else method_name

fit <- gam(
  formula = as.formula(formula_text),
  data = data,
  family = family_obj,
  method = fit_method,
  select = select_flag
)

prefit_args <- list(
  formula = as.formula(formula_text),
  data = data,
  family = family_obj,
  method = fit_method,
  fit = FALSE,
  select = select_flag
)
if (length(fit$sp) > 0L) {
  prefit_args$sp <- fit$sp
}
G <- do.call(gam, prefit_args)

G$family <- mgcv:::fix.family(G$family)
G$rS <- mgcv:::mini.roots(G$S, G$off, ncol(G$X), G$rank)
Ssp <- mgcv:::totalPenaltySpace(G$S, G$H, G$off, ncol(G$X))
G$Eb <- Ssp$E
G$U1 <- cbind(Ssp$Y, Ssp$Z)
G$Mp <- ncol(Ssp$Z)
G$UrS <- list()
if (length(G$S) > 0) {
  for (i in seq_along(G$S)) {
    G$UrS[[i]] <- t(Ssp$Y) %*% G$rS[[i]]
  }
} else {
  i <- 0
}
if (!is.null(G$H)) {
  G$UrS[[i + 1]] <- t(Ssp$Y) %*% mgcv:::mroot(G$H)
}

if (length(G$off) == 0L) {
  log_sp_full <- numeric(0)
} else if (length(G$sp) == 0L) {
  log_sp_full <- unname(as.numeric(G$lsp0))
} else if (is.null(G$L)) {
  log_sp_full <- unname(as.numeric(log(G$sp) + G$lsp0))
} else {
  log_sp_full <- unname(as.numeric(G$L %*% log(G$sp) + G$lsp0))
}

rp <- mgcv:::gam.reparam(G$UrS, lsp = log_sp_full, deriv = 2)

payload <- list(
  fit_sp = serialize_numeric(fit$sp),
  log_sp_full = serialize_numeric(log_sp_full),
  setup = list(
    E = unname(Ssp$E),
    Eb = unname(G$Eb),
    U1 = unname(G$U1),
    UrS = lapply(G$UrS, function(M) unname(M)),
    Mp = as.integer(G$Mp)
  ),
  gam_reparam = list(
    S = unname(rp$S),
    E = unname(rp$E),
    Qs = unname(rp$Qs),
    rS = lapply(rp$rS, function(M) unname(M)),
    det = unname(as.numeric(rp$det)),
    det1 = serialize_numeric(rp$det1),
    det2 = if (is.null(rp$det2)) NULL else unname(rp$det2),
    fixed_penalty = isTRUE(rp$fixed.penalty)
  )
)

write_json(
  payload,
  output_json,
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
