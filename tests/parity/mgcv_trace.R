# Usage:
#   Rscript mgcv_trace.R <csv_path> <output_json> <formula> <family> <method> <select>
#
# Fits mgcv::gam and writes trace of optimization for parity with
# nampy.gam.fit.gam_trace.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop("Usage: Rscript mgcv_trace.R <csv_path> <output_json> <formula> <family> <method> <select>")
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
method_name <- args[[5]]
select_flag <- tolower(args[[6]]) %in% c("true", "1", "yes")

suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))

data <- read.csv(csv_path, stringsAsFactors = FALSE)
for (nm in names(data)) {
  if (is.character(data[[nm]])) data[[nm]] <- factor(data[[nm]])
}
family_obj <- switch(
  family_name,
  gaussian = gaussian(),
  binomial = binomial(link = "logit"),
  poisson = poisson(link = "log"),
  stop(sprintf("Unsupported family for trace parity: %s", family_name))
)

fit <- gam(
  formula = as.formula(formula_text),
  data = data,
  family = family_obj,
  method = method_name,
  select = select_flag
)

outer_info <- fit$outer.info
score_hist <- NULL
if (!is.null(outer_info) && !is.null(outer_info$score.hist)) {
  score_hist <- unname(as.numeric(outer_info$score.hist))
}

trace_rows <- list()
if (!is.null(score_hist) && length(score_hist) > 0) {
  for (i in seq_along(score_hist)) {
    trace_rows[[length(trace_rows) + 1]] <- list(
      iter = as.integer(i),
      log_sp = NULL,
      criterion = score_hist[[i]],
      gradient = NULL,
      hessian = NULL,
      accepted_step_norm = NULL,
      rank_info = NULL
    )
  }
}

trace_rows[[length(trace_rows) + 1]] <- list(
  iter = as.integer(length(trace_rows) + 1),
  log_sp = unname(as.numeric(log(fit$sp))),
  criterion = unname(as.numeric(fit$gcv.ubre)),
  gradient = if (!is.null(outer_info) && !is.null(outer_info$grad)) unname(as.numeric(outer_info$grad)) else NULL,
  hessian = if (!is.null(outer_info) && !is.null(outer_info$hess)) unname(as.matrix(outer_info$hess)) else NULL,
  accepted_step_norm = 0.0,
  rank_info = NULL
)

payload <- list(
  fit = list(
    criterion_name = method_name,
    smoothing_params = unname(as.numeric(fit$sp)),
    outer_info = list(
      conv = if (!is.null(outer_info) && !is.null(outer_info$conv)) as.character(outer_info$conv) else NULL,
      iter = if (!is.null(outer_info) && !is.null(outer_info$iter)) as.integer(outer_info$iter) else NULL,
      score_hist = score_hist
    )
  ),
  trace = trace_rows
)

write_json(payload, output_json, auto_unbox = TRUE, digits = 17, pretty = TRUE)
