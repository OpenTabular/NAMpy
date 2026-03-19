args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop("Usage: Rscript mgcv_snapshot.R <csv_path> <output_json> <formula> <family> <method> <select>")
}

csv_path <- args[[1]]
output_json <- args[[2]]
formula_text <- args[[3]]
family_name <- tolower(args[[4]])
method_name <- args[[5]]
select_flag <- tolower(args[[6]]) %in% c("true", "1", "yes")

suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))

data <- read.csv(csv_path, stringsAsFactors = FALSE)
response_name <- all.vars(as.formula(formula_text))[1]

family_parts <- strsplit(family_name, ":", fixed = TRUE)[[1]]
family_key <- family_parts[[1]]
family_param <- if (length(family_parts) >= 2) family_parts[[2]] else NULL

family_obj <- switch(
  family_key,
  gaussian = gaussian(),
  binomial = binomial(link = "logit"),
  poisson = poisson(link = "log"),
  gamma = Gamma(link = "log"),
  negbin = {
    theta <- if (is.null(family_param)) 1.0 else as.numeric(family_param)
    mgcv::nb(theta = theta, link = "log")
  },
  stop(sprintf("Unsupported family for parity snapshot: %s", family_name))
)

fit <- gam(
  formula = as.formula(formula_text),
  data = data,
  family = family_obj,
  method = method_name,
  select = select_flag
)

pred_response <- unname(as.numeric(predict(fit, type = "response")))
pred_link <- unname(as.numeric(predict(fit, type = "link")))
pred_terms <- unname(as.matrix(predict(fit, type = "terms")))
pred_lpmatrix <- unname(as.matrix(predict(fit, type = "lpmatrix")))

coef_full <- unname(as.numeric(coef(fit)))
coef_names <- names(coef(fit))
intercept <- if ("(Intercept)" %in% coef_names) {
  unname(as.numeric(coef(fit)["(Intercept)"]))
} else {
  0.0
}

edf_by_term <- unname(as.numeric(summary(fit)$edf))
edf_total <- unname(as.numeric(sum(fit$edf)))
trace_H <- edf_total
scale_val <- unname(as.numeric(fit$sig2))
rss_val <- if (family_name == "gaussian") {
  y <- as.numeric(data[[response_name]])
  unname(as.numeric(sum((y - pred_response) ^ 2)))
} else {
  NULL
}

snapshot <- list(
  fit = list(
    family_name = family_name,
    link_name = fit$family$link,
    criterion_name = method_name,
    criterion_value = unname(as.numeric(fit$gcv.ubre)),
    coef_full = coef_full,
    intercept = intercept,
    smoothing_params = unname(as.numeric(fit$sp)),
    edf_total = edf_total,
    edf_by_term = edf_by_term,
    trace_H = trace_H,
    scale = scale_val,
    rss = rss_val,
    deviance = unname(as.numeric(fit$deviance))
  ),
  predictions = list(
    response = pred_response,
    link = pred_link,
    terms = pred_terms,
    lpmatrix = pred_lpmatrix
  )
)

write_json(snapshot, output_json, auto_unbox = TRUE, digits = 17, pretty = TRUE)
