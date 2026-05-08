# Usage:
#   Rscript mgcv_fixed_sp_fit3.R <csv_path> <output_json> <formula> <family> <sp_json> [score_type]
#
# Builds mgcv's `estimate.gam` setup, then calls low-level `gam.fit3()` at fixed
# smoothing parameters and writes exposed coefficient/PIRLS/derivative state.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop(
    "Usage: Rscript mgcv_fixed_sp_fit3.R <csv_path> <output_json> <formula> <family> <sp_json> [score_type]"
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

csv_path <- args[[1]]
output_json <- args[[2]]
formula_text <- normalize_formula_text(args[[3]])
family_name <- tolower(args[[4]])
sp <- as.numeric(jsonlite::fromJSON(args[[5]]))
score_type <- if (length(args) >= 6) toupper(args[[6]]) else "REML"

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

family_obj <- switch(
  family_key,
  gaussian = gaussian(),
  poisson = poisson(link = "log"),
  binomial = binomial(link = "logit"),
  Gamma = Gamma(link = "inverse"),
  gamma = Gamma(link = "inverse"),
  stop(sprintf("Unsupported family for gam.fit3 fixed-sp parity: %s", family_name))
)
family_obj <- mgcv:::fix.family.link(family_obj)
family_obj <- mgcv:::fix.family.var(family_obj)
family_obj <- mgcv:::fix.family.ls(family_obj)

G <- gam(
  formula = as.formula(formula_text),
  data = data,
  family = family_obj,
  method = "REML",
  sp = unname(sp),
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

fit <- mgcv:::gam.fit3(
  x = G$X,
  y = G$y,
  sp = log(pmax(unname(sp), 1e-300)),
  Eb = G$Eb,
  UrS = G$UrS,
  weights = G$w,
  offset = G$offset,
  U1 = G$U1,
  Mp = G$Mp,
  family = G$family,
  control = gam.control(),
  intercept = TRUE,
  deriv = 2,
  gamma = 1,
  scale = 1,
  scoreType = score_type,
  null.coef = rep(0, ncol(G$X)),
  pearson.extra = G$pearson.extra,
  dev.extra = G$dev.extra,
  n.true = G$n.true
)

num_or_null <- function(x) {
  if (is.null(x)) return(NULL)
  unname(as.numeric(x))
}

mat_or_null <- function(x) {
  if (is.null(x)) return(NULL)
  unname(as.matrix(x))
}

payload <- list(
  coefficients = unname(as.numeric(fit$coefficients)),
  linear_predictors = unname(as.numeric(fit$linear.predictors)),
  fitted_values = unname(as.numeric(fit$fitted.values)),
  deviance = unname(as.numeric(fit$deviance)),
  rV = unname(as.matrix(fit$rV)),
  K = unname(as.matrix(fit$K)),
  scale_est = unname(as.numeric(fit$scale.est)),
  reml_scale = if (is.na(fit$reml.scale)) NULL else unname(as.numeric(fit$reml.scale)),
  weights = unname(as.numeric(fit$weights)),
  working_weights = unname(as.numeric(fit$working.weights)),
  prior_weights = unname(as.numeric(fit$prior.weights)),
  working_response = unname(as.numeric(fit$z)),
  REML = num_or_null(fit$REML),
  REML1 = num_or_null(fit$REML1),
  REML2 = mat_or_null(fit$REML2),
  GCV = num_or_null(fit$GCV),
  GCV1 = num_or_null(fit$GCV1),
  GCV2 = mat_or_null(fit$GCV2),
  UBRE = num_or_null(fit$UBRE),
  UBRE1 = num_or_null(fit$UBRE1),
  UBRE2 = mat_or_null(fit$UBRE2),
  D1 = num_or_null(fit$D1),
  D2 = mat_or_null(fit$D2),
  P = num_or_null(fit$P),
  P1 = num_or_null(fit$P1),
  P2 = mat_or_null(fit$P2),
  trA = num_or_null(fit$trA),
  trA1 = num_or_null(fit$trA1),
  trA2 = mat_or_null(fit$trA2),
  db_drho = mat_or_null(fit$db.drho),
  dVkk = mat_or_null(fit$dVkk),
  ldetS1 = num_or_null(fit$ldetS1)
)

write_json(
  payload,
  output_json,
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
