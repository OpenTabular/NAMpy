# Usage:
#   Rscript mgcv_fixed_sp_magic.R <csv_path> <output_json> <formula> <sp_json>
#
# Runs mgcv's Gaussian fixed-sp fit through the `magic` backend and records
# coefficient/state outputs plus the fixed-sp REML score at the same smoothing
# parameters.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: Rscript mgcv_fixed_sp_magic.R <csv_path> <output_json> <formula> <sp_json>")
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
sp <- as.numeric(jsonlite::fromJSON(args[[4]]))

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

formula_obj <- as.formula(formula_text)
response_name <- all.vars(formula_obj[[2]])[[1]]

magic_fit_args <- list(
  formula = formula_obj,
  data = data,
  family = gaussian(),
  method = "GCV.Cp",
  sp = unname(sp)
)

reml_fit_args <- list(
  formula = formula_obj,
  data = data,
  family = gaussian(),
  method = "REML",
  sp = unname(sp)
)

fit <- do.call(mgcv::gam, magic_fit_args)
reml_fit <- do.call(mgcv::gam, reml_fit_args)

offset_vec <- if (is.null(fit$offset)) {
  rep(0.0, nrow(data))
} else {
  as.numeric(fit$offset)
}

working_response <- as.numeric(data[[response_name]]) - offset_vec

payload <- list(
  coefficients = unname(as.numeric(fit$coefficients)),
  linear_predictors = unname(as.numeric(fit$linear.predictors)),
  fitted_values = unname(as.numeric(fit$fitted.values)),
  deviance = unname(as.numeric(fit$deviance)),
  prior_weights = unname(as.numeric(fit$prior.weights)),
  weights = unname(as.numeric(fit$weights)),
  working_weights = unname(as.numeric(fit$weights)),
  working_response = unname(as.numeric(working_response)),
  reml = unname(as.numeric(reml_fit$gcv.ubre))
)

write_json(
  payload,
  output_json,
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
