# Usage:
#   Rscript mgcv_negbin_inner_trace.R <csv_path> <output_json> <formula> <family>
#
# Build an mgcv prefit object, then call the low-level gam.fit3/gam.fit4
# fixed-sp path directly while wrapping family$putTheta so the actual inner
# theta updates are recorded.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: Rscript mgcv_negbin_inner_trace.R <csv_path> <output_json> <formula> <family>")
}

normalize_formula_text <- function(x) {
  x <- gsub("\\[", "c(", x)
  x <- gsub("\\]", ")", x)
  x <- gsub("\\bTrue\\b", "TRUE", x)
  x <- gsub("\\bFalse\\b", "FALSE", x)
  x <- gsub("\\bNone\\b", "NULL", x)
  x
}

csv_path <- args[[1]]
output_json <- args[[2]]
formula_text <- normalize_formula_text(args[[3]])
family_name <- tolower(args[[4]])

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
  negbin_est = {
    theta <- if (is.null(family_param)) 1.0 else as.numeric(family_param)
    mgcv::nb(theta = -abs(theta), link = "log")
  },
  negbin = {
    theta <- if (is.null(family_param)) 1.0 else as.numeric(family_param)
    mgcv::nb(theta = theta, link = "log")
  },
  stop(sprintf("Unsupported family for negbin inner trace: %s", family_name))
)

family_obj <- mgcv:::fix.family.link(family_obj)
family_obj <- mgcv:::fix.family.var(family_obj)
family_obj <- mgcv:::fix.family.ls(family_obj)

G <- gam(
  formula = as.formula(formula_text),
  data = data,
  family = family_obj,
  method = "REML",
  sp = 1,
  fit = FALSE
)

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
}

trace_env <- new.env(parent = emptyenv())
trace_env$theta_trace <- numeric(0)
orig_put_theta <- G$family$putTheta
G$family$putTheta <- function(theta) {
  trace_env$theta_trace <- c(trace_env$theta_trace, as.numeric(theta[[1]]))
  orig_put_theta(theta)
}

fit <- mgcv:::gam.fit3(
  x = G$X,
  y = G$y,
  sp = c(log(abs(as.numeric(family_param))), log(1)),
  Eb = G$Eb,
  UrS = G$UrS,
  weights = G$w,
  offset = G$offset,
  U1 = G$U1,
  Mp = G$Mp,
  family = G$family,
  control = gam.control(),
  intercept = TRUE,
  deriv = 0,
  gamma = 1,
  scale = 0,
  scoreType = "EFS",
  null.coef = rep(0, ncol(G$X)),
  pearson.extra = G$pearson.extra,
  dev.extra = G$dev.extra,
  n.true = G$n.true
)

payload <- list(
  fit = list(
    smoothing_params = 1,
    family_theta = unname(as.numeric(fit$family$getTheta(TRUE)))
  ),
  inner_trace = lapply(
    seq_along(trace_env$theta_trace),
    function(i) list(
      iter = as.integer(i),
      log_theta = as.numeric(trace_env$theta_trace[[i]])
    )
  )
)

write_json(payload, output_json, auto_unbox = TRUE, digits = 17, pretty = TRUE)
