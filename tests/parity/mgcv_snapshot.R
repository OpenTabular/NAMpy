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

normalize_smooth_label <- function(x) {
  x <- as.character(x)
  gsub(",\\s*k\\s*=\\s*[^,)]+", "", x)
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
  binomial = {
    link <- if (is.null(family_param) || family_param == "") "logit" else family_param
    binomial(link = link)
  },
  poisson = poisson(link = "log"),
  gamma = {
    link <- if (is.null(family_param) || family_param == "") "log" else family_param
    Gamma(link = link)
  },
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

fixed_sp_derivatives <- function(sp_ref, eps_grad = 1e-6, eps_hess = 1e-4) {
  sp_ref <- as.numeric(sp_ref)
  if (length(sp_ref) == 0) {
    return(list(grad = numeric(0), hess = matrix(numeric(0), 0, 0)))
  }
  log_sp_ref <- log(pmax(sp_ref, 1e-300))

  eval_at_log_sp <- function(log_sp) {
    fixed_args <- gam_args
    fixed_args$sp <- unname(exp(log_sp))
    fixed_fit <- do.call(gam, fixed_args)
    unname(as.numeric(fixed_fit$gcv.ubre))
  }

  grad_at_log_sp <- function(log_sp) {
    g <- rep(NA_real_, length(log_sp))
    steps <- pmax(eps_grad, 1e-5 * (1 + abs(log_sp)))
    for (i in seq_along(log_sp)) {
      plus1 <- log_sp
      minus1 <- log_sp
      plus2 <- log_sp
      minus2 <- log_sp
      plus1[i] <- plus1[i] + steps[i]
      minus1[i] <- minus1[i] - steps[i]
      plus2[i] <- plus2[i] + 2 * steps[i]
      minus2[i] <- minus2[i] - 2 * steps[i]
      g[i] <- (
        -eval_at_log_sp(plus2) +
          8 * eval_at_log_sp(plus1) -
          8 * eval_at_log_sp(minus1) +
          eval_at_log_sp(minus2)
      ) / (12 * steps[i])
    }
    unname(g)
  }

  grad <- grad_at_log_sp(log_sp_ref)
  hess <- matrix(0.0, length(log_sp_ref), length(log_sp_ref))
  steps_h <- pmax(eps_hess, 1e-3 * (1 + abs(log_sp_ref)))
  f0 <- eval_at_log_sp(log_sp_ref)
  for (j in seq_along(log_sp_ref)) {
    for (k in j:length(log_sp_ref)) {
      if (j == k) {
        plus1 <- log_sp_ref
        minus1 <- log_sp_ref
        plus2 <- log_sp_ref
        minus2 <- log_sp_ref
        plus1[j] <- plus1[j] + steps_h[j]
        minus1[j] <- minus1[j] - steps_h[j]
        plus2[j] <- plus2[j] + 2 * steps_h[j]
        minus2[j] <- minus2[j] - 2 * steps_h[j]
        hess[j, j] <- (
          -eval_at_log_sp(plus2) +
            16 * eval_at_log_sp(plus1) -
            30 * f0 +
            16 * eval_at_log_sp(minus1) -
            eval_at_log_sp(minus2)
        ) / (12 * steps_h[j] * steps_h[j])
      } else {
        pp <- log_sp_ref
        pm <- log_sp_ref
        mp <- log_sp_ref
        mm <- log_sp_ref
        pp[j] <- pp[j] + steps_h[j]
        pp[k] <- pp[k] + steps_h[k]
        pm[j] <- pm[j] + steps_h[j]
        pm[k] <- pm[k] - steps_h[k]
        mp[j] <- mp[j] - steps_h[j]
        mp[k] <- mp[k] + steps_h[k]
        mm[j] <- mm[j] - steps_h[j]
        mm[k] <- mm[k] - steps_h[k]
        hess[j, k] <- (
          eval_at_log_sp(pp) -
            eval_at_log_sp(pm) -
            eval_at_log_sp(mp) +
            eval_at_log_sp(mm)
        ) / (4 * steps_h[j] * steps_h[k])
        hess[k, j] <- hess[j, k]
      }
    }
  }
  list(grad = grad, hess = hess)
}

fixed_outer <- fixed_sp_derivatives(fit$sp)

pred_response <- unname(as.numeric(predict(fit, type = "response")))
pred_link <- unname(as.numeric(predict(fit, type = "link")))
pred_terms <- unname(as.matrix(predict(fit, type = "terms")))
pred_term_names <- colnames(predict(fit, type = "terms"))
pred_lpmatrix <- unname(as.matrix(predict(fit, type = "lpmatrix")))
pred_se_response <- tryCatch(
  unname(as.numeric(predict(fit, type = "response", se.fit = TRUE)$se.fit)),
  error = function(e) NULL
)
pred_se_link <- tryCatch(
  unname(as.numeric(predict(fit, type = "link", se.fit = TRUE)$se.fit)),
  error = function(e) NULL
)

conc_full <- tryCatch(concurvity(fit, full = TRUE), error = function(e) NULL)
sp_cov <- tryCatch(sp.vcov(fit, edge.correct = FALSE), error = function(e) NULL)
gam_vc <- tryCatch(gam.vcomp(fit, rescale = FALSE), error = function(e) NULL)
residuals_block <- list(
  response = unname(as.numeric(residuals(fit, type = "response"))),
  working = unname(as.numeric(residuals(fit, type = "working"))),
  pearson = unname(as.numeric(residuals(fit, type = "pearson"))),
  scaled_pearson = unname(as.numeric(residuals(fit, type = "scaled.pearson"))),
  deviance = unname(as.numeric(residuals(fit, type = "deviance")))
)
k_check_table <- tryCatch({
  set.seed(0)
  out <- k.check(fit, subsample = 120, n.rep = 8)
  list(
    labels = unname(as.character(rownames(out))),
    values = unname(as.matrix(out))
  )
}, error = function(e) NULL)
one_se <- NULL
if (!is.null(sp_cov) && length(fit$sp) > 0) {
  d <- sqrt(diag(sp_cov))
  if (length(d) > 0 && all(is.finite(d)) && all(d > 0)) {
    alpha <- sqrt(2 * length(d)) / as.numeric(t(d) %*% solve(sp_cov, d))
    lsp <- log(as.numeric(fit$sp))
    lsp <- lsp + alpha * d
    one_se <- unname(exp(lsp))
  }
}

anova_single <- tryCatch(anova(fit, freq = FALSE), error = function(e) NULL)
anova_parametric <- NULL
anova_smooth <- NULL
if (!is.null(anova_single)) {
  if (!is.null(anova_single$pTerms.table)) {
    anova_parametric <- list(
      labels = unname(as.character(rownames(anova_single$pTerms.table))),
      values = unname(as.matrix(anova_single$pTerms.table))
    )
  }
  if (!is.null(anova_single$s.table)) {
    anova_smooth <- list(
      labels = unname(as.character(rownames(anova_single$s.table))),
      values = unname(as.matrix(anova_single$s.table))
    )
  }
}

smooth_cov_bayes <- NULL
smooth_cov_freq <- NULL
smooth_edf1 <- NULL
smooth_test_inputs <- NULL
smooth_function_space <- NULL
if (length(fit$smooth) > 0) {
  bayes_blocks <- list()
  freq_blocks <- list()
  edf1_vals <- c()
  labels <- c()
  coef_blocks <- list()
  r_blocks <- list()
  edf_vals <- c()
  fitted_blocks <- list()
  var_diag_blocks <- list()
  for (sm in fit$smooth) {
    ind <- sm$first.para:sm$last.para
    labels <- c(labels, sm$label)
    bayes_blocks[[length(bayes_blocks) + 1]] <- unname(fit$Vp[ind, ind, drop = FALSE])
    if (!is.null(fit$Ve)) {
      freq_blocks[[length(freq_blocks) + 1]] <- unname(fit$Ve[ind, ind, drop = FALSE])
    }
    coef_blocks[[length(coef_blocks) + 1]] <- unname(as.numeric(coef(fit)[ind]))
    term_col <- which(normalize_smooth_label(pred_term_names) == normalize_smooth_label(sm$label))
    if (length(term_col) != 1) {
      stop(sprintf("mgcv_snapshot.R: could not uniquely match smooth label %s in predict(type='terms').", sm$label))
    }
    fitted_blocks[[length(fitted_blocks) + 1]] <- unname(as.numeric(pred_terms[, term_col]))
    Xi <- pred_lpmatrix[, ind, drop = FALSE]
    var_diag_blocks[[length(var_diag_blocks) + 1]] <- unname(as.numeric(rowSums((Xi %*% fit$Vp[ind, ind, drop = FALSE]) * Xi)))
    if (!is.null(fit$R)) {
      r_blocks[[length(r_blocks) + 1]] <- unname(as.matrix(fit$R[, ind, drop = FALSE]))
    }
    edf_vals <- c(edf_vals, sum(fit$edf[ind]))
    if (!is.null(fit$edf1)) {
      edf1_vals <- c(edf1_vals, sum(fit$edf1[ind]))
    } else {
      edf1_vals <- c(edf1_vals, NA_real_)
    }
  }
  smooth_cov_bayes <- list(labels = unname(as.character(labels)), blocks = bayes_blocks)
  smooth_cov_freq <- if (length(freq_blocks) == length(labels)) {
    list(labels = unname(as.character(labels)), blocks = freq_blocks)
  } else {
    NULL
  }
  smooth_edf1 <- list(labels = unname(as.character(labels)), values = unname(as.numeric(edf1_vals)))
  smooth_test_inputs <- list(
    labels = unname(as.character(labels)),
    coef_blocks = coef_blocks,
    r_blocks = if (length(r_blocks) == length(labels)) r_blocks else NULL,
    edf = unname(as.numeric(edf_vals)),
    edf1 = unname(as.numeric(edf1_vals)),
    residual_df = unname(as.numeric(anova_single$residual.df))
  )
  smooth_function_space <- list(
    labels = unname(as.character(labels)),
    fitted = fitted_blocks,
    variance_diag = var_diag_blocks
  )
}

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
    log_smoothing_params = unname(as.numeric(log(pmax(fit$sp, 1e-300)))),
    edf_total = edf_total,
    edf_by_term = edf_by_term,
    trace_H = trace_H,
    scale = scale_val,
    rss = rss_val,
    deviance = unname(as.numeric(fit$deviance)),
    cov_bayes = if (is.null(fit$Vp)) NULL else unname(fit$Vp),
    cov_freq = if (is.null(fit$Ve)) NULL else unname(fit$Ve),
    dev_sum_dev_resids = unname(as.numeric(dev_sum_dev_resids)),
    penalty_quadratic = unname(as.numeric(penalty_quadratic)),
    n_obs = nrow(data),
    outer_grad = unname(as.numeric(fixed_outer$grad)),
    outer_hess = unname(as.matrix(fixed_outer$hess))
  ),
  predictions = list(
    response = pred_response,
    link = pred_link,
    terms = pred_terms,
    lpmatrix = pred_lpmatrix,
    se_response = pred_se_response,
    se_link = pred_se_link
  ),
  parity = list(
    diagnostics = list(
      concurvity_labels = if (is.null(conc_full)) NULL else colnames(conc_full),
      concurvity_full = if (is.null(conc_full)) NULL else unname(conc_full),
      sp_vcov = if (is.null(sp_cov)) NULL else unname(sp_cov),
      gam_vcomp = if (is.null(gam_vc) || is.null(gam_vc$vc)) NULL else unname(gam_vc$vc),
      gam_vcomp_names = if (is.null(gam_vc) || is.null(gam_vc$vc)) NULL else rownames(gam_vc$vc),
      one_se_rule = one_se,
      residuals = residuals_block,
      k_check = k_check_table,
      anova_parametric = anova_parametric,
      anova_smooth = anova_smooth,
      smooth_cov_bayes = smooth_cov_bayes,
      smooth_cov_freq = smooth_cov_freq,
      smooth_edf1 = smooth_edf1,
      smooth_test_inputs = smooth_test_inputs,
      smooth_function_space = smooth_function_space
    )
  )
)

write_json(snapshot, output_json, auto_unbox = TRUE, digits = 17, pretty = TRUE)
