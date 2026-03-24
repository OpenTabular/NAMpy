# Usage:
#   Rscript mgcv_gaussian_smoothness_postprocess.R <csv_path> <output_json> <formula> <method> <sp_json>
#
# Fits mgcv::gam at fixed linear smoothing parameters sp (JSON array) and writes
# REML / GCV criterion value, sum(edf), and deviance scale for parity with
# nampy.gam.fit.postprocess.gaussian_smoothness_postprocess.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop("Usage: Rscript mgcv_gaussian_smoothness_postprocess.R <csv> <out.json> <formula> <method> <sp_json>")
}

csv_path <- args[[1]]
output_json <- args[[2]]
formula_text <- args[[3]]
method_name <- toupper(args[[4]])
sp_json <- args[[5]]

mgcv_lib <- Sys.getenv("MGCV_LIB_PATH", "")
if (nzchar(mgcv_lib)) {
  .libPaths(c(mgcv_lib, .libPaths()))
}

suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))

normalize_formula_text <- function(x) {
  x <- gsub("\\[", "c(", x)
  x <- gsub("\\]", ")", x)
  x <- gsub("\\bTrue\\b", "TRUE", x)
  x <- gsub("\\bFalse\\b", "FALSE", x)
  x <- gsub("\\bNone\\b", "NULL", x)
  x
}

formula_text <- normalize_formula_text(formula_text)
data <- read.csv(csv_path, stringsAsFactors = FALSE)
for (nm in names(data)) {
  if (is.character(data[[nm]])) data[[nm]] <- factor(data[[nm]])
}

sp_vec <- as.numeric(fromJSON(sp_json))
if (any(!is.finite(sp_vec))) {
  stop("Non-finite smoothing parameters in sp_json")
}

fit_method <- if (method_name %in% c("GCV", "GACV")) "GCV.Cp" else "REML"

gam_args <- list(
  formula = as.formula(formula_text),
  data = data,
  family = gaussian(),
  method = fit_method,
  sp = sp_vec
)
fit <- do.call(gam, gam_args)

tr_a <- sum(fit$edf)
score <- unname(as.numeric(fit$gcv.ubre))
sig2 <- unname(as.numeric(fit$sig2))
dev <- unname(as.numeric(fit$deviance))

out <- list(
  tr_a = tr_a,
  criterion_value = score,
  scale_est = sig2,
  deviance = dev,
  method = fit_method
)

write_json(out, output_json, auto_unbox = TRUE, digits = 17, pretty = TRUE)
