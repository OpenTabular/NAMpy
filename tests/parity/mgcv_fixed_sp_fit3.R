# Usage:
#   Rscript mgcv_fixed_sp_fit3.R <csv_path> <output_json> <formula> <family> <sp_json>
#
# Builds mgcv's `estimate.gam` setup, then calls low-level `gam.fit3()` at fixed
# smoothing parameters and writes exposed coefficient/PIRLS/derivative state.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop(
    "Usage: Rscript mgcv_fixed_sp_fit3.R <csv_path> <output_json> <formula> <family> <sp_json>"
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
  Gamma = Gamma(link = "log"),
  gamma = Gamma(link = "log"),
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
  scoreType = "REML",
  null.coef = rep(0, ncol(G$X)),
  pearson.extra = G$pearson.extra,
  dev.extra = G$dev.extra,
  n.true = G$n.true
)

payload <- list(
  coefficients = unname(as.numeric(fit$coefficients)),
  linear_predictors = unname(as.numeric(fit$linear.predictors)),
  fitted_values = unname(as.numeric(fit$fitted.values)),
  deviance = unname(as.numeric(fit$deviance)),
  weights = unname(as.numeric(fit$weights)),
  working_weights = unname(as.numeric(fit$working.weights)),
  prior_weights = unname(as.numeric(fit$prior.weights)),
  working_response = unname(as.numeric(fit$z)),
  REML = unname(as.numeric(fit$REML)),
  REML1 = unname(as.numeric(fit$REML1)),
  REML2 = unname(as.matrix(fit$REML2)),
  D1 = unname(as.numeric(fit$D1)),
  D2 = unname(as.matrix(fit$D2)),
  P = unname(as.numeric(fit$P)),
  P1 = unname(as.numeric(fit$P1)),
  P2 = unname(as.matrix(fit$P2)),
  trA = unname(as.numeric(fit$trA)),
  trA1 = unname(as.numeric(fit$trA1)),
  trA2 = unname(as.matrix(fit$trA2)),
  db_drho = unname(as.matrix(fit$db.drho)),
  dVkk = unname(as.matrix(fit$dVkk)),
  ldetS1 = unname(as.numeric(fit$ldetS1))
)

write_json(
  payload,
  output_json,
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
