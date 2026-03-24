# Usage:
#   Rscript mgcv_snapshot.R <csv_path> <output_json> <formula> <family> <method> <select>
#
# Fits mgcv::gam at fixed linear smoothing parameters sp (JSON array) and writes
# REML / GCV criterion value, sum(edf), and deviance scale for parity with
# nampy.gam.fit.postprocess.gaussian_smoothness_postprocess.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop("Usage: Rscript mgcv_snapshot.R <csv_path> <output_json> <formula> <family> <method> <select>")
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
fit_method <- if (tolower(method_name) == "fixed") "REML" else method_name

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

gam_args <- list(
  formula = as.formula(formula_text),
  data = data,
  family = family_obj,
  method = fit_method,
  select = select_flag
)
if (length(args) >= 7) {
  wcol <- args[[7]]
  if (nzchar(wcol) && tolower(wcol) != "none" && wcol != "-") {
    if (!wcol %in% names(data)) {
      stop(sprintf("mgcv_snapshot.R: weights column %s not found in CSV.", wcol))
    }
    gam_args$weights <- data[[wcol]]
  }
}
fit <- do.call(gam, gam_args)

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
prior_w <- fit$prior.weights
if (is.null(prior_w)) {
  prior_w <- rep(1, nrow(data))
}
prior_w <- as.numeric(prior_w)

rss_val <- if (family_name == "gaussian") {
  y <- as.numeric(data[[response_name]])
  unname(as.numeric(sum(prior_w * (y - pred_response) ^ 2)))
} else {
  NULL
}
y_obs <- as.numeric(data[[response_name]])
mu_fit <- pred_response
dev_sum_dev_resids <- sum(fit$family$dev.resids(y_obs, mu_fit, prior_w))

penalty_quadratic <- 0.0
if (length(fit$smooth) > 0) {
  cf <- coef(fit)
  for (sm in fit$smooth) {
    beta <- cf[sm$first.para:sm$last.para]
    isp <- sm$first.sp:sm$last.sp
    lam <- fit$sp[isp]
    if (length(lam) != length(sm$S)) {
      stop(
        sprintf(
          "mgcv_snapshot.R: penalty lam/S length mismatch for smooth %s: len(lam)=%d len(S)=%d",
          sm$label,
          length(lam),
          length(sm$S)
        )
      )
    }
    for (k in seq_along(sm$S)) {
      Sk <- sm$S[[k]]
      penalty_quadratic <- penalty_quadratic + lam[k] * as.numeric(t(beta) %*% Sk %*% beta)
    }
  }
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
    deviance = unname(as.numeric(fit$deviance)),
    dev_sum_dev_resids = unname(as.numeric(dev_sum_dev_resids)),
    penalty_quadratic = unname(as.numeric(penalty_quadratic)),
    n_obs = nrow(data)
  ),
  predictions = list(
    response = pred_response,
    link = pred_link,
    terms = pred_terms,
    lpmatrix = pred_lpmatrix
  )
)

write_json(snapshot, output_json, auto_unbox = TRUE, digits = 17, pretty = TRUE)
