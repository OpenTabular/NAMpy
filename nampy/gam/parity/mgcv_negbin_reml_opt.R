suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(mgcv))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop("Usage: Rscript mgcv_negbin_reml_opt.R <csv> <json> <formula> <theta0> <method>")
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
json_path <- args[[2]]
formula_text <- normalize_formula_text(args[[3]])
theta0 <- as.numeric(args[[4]])
method <- toupper(args[[5]])

data <- read.csv(csv_path, stringsAsFactors = FALSE)
for (nm in names(data)) {
  if (is.character(data[[nm]])) data[[nm]] <- factor(data[[nm]])
}

fit <- gam(
  formula = as.formula(formula_text),
  data = data,
  family = mgcv::nb(theta = -abs(theta0), link = "log"),
  method = method
)

payload <- list(
  smoothing_params = unname(as.numeric(fit$sp)),
  family_theta = unname(as.numeric(fit$family$getTheta(TRUE))),
  criterion_value = unname(as.numeric(fit$gcv.ubre)),
  outer_iter = if (is.null(fit$outer.info$iter)) 0L else as.integer(fit$outer.info$iter)
)

write_json(payload, json_path, auto_unbox = TRUE, digits = 17, pretty = TRUE)
