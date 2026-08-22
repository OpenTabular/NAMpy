# Usage:
#   Rscript debug/gamma_bfgs_initial_probe.R <csv_path> <formula>
#
# Captures the initial state inside mgcv:::bfgs() for the Gamma REML BFGS
# parity case.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript gamma_bfgs_initial_probe.R <csv_path> <formula>")

find_paths <- function(expr, pattern, path = integer()) {
  hits <- list()
  txt <- paste(deparse(expr, width.cutoff = 500L), collapse = " ")
  if (grepl(pattern, txt, fixed = TRUE)) hits[[length(hits) + 1L]] <- path
  if (is.call(expr) || is.pairlist(expr) || is.expression(expr)) {
    parts <- as.list(expr)
    for (i in seq_along(parts)) hits <- c(hits, find_paths(parts[[i]], pattern, c(path, i)))
  }
  hits
}

deepest_path <- function(fun, pattern) {
  hits <- Filter(length, find_paths(body(fun), pattern))
  if (length(hits) == 0L) stop(sprintf("Failed to locate pattern: %s", pattern))
  hits[[which.max(vapply(hits, length, integer(1)))]]
}

suppressPackageStartupMessages(library(mgcv))
options(digits = 17)
print(deparse(Gamma(link = "inverse")$initialize))

csv_path <- args[[1]]
formula_text <- args[[2]]
data <- read.csv(csv_path, stringsAsFactors = FALSE)

probe_env <- new.env(parent = emptyenv())
probe_env$initial <- NULL
probe_env$inverse_hessian <- NULL
probe_env$fits <- list()
probe_env$inner_rows <- list()

gam_fit3_fun <- get("gam.fit3", envir = asNamespace("mgcv"))
trace(
  "gam.fit3",
  where = asNamespace("mgcv"),
  print = FALSE,
  at = list(deepest_path(gam_fit3_fun, "div.thresh <- 10 *")),
  tracer = quote({
    probe_env$inner_rows[[length(probe_env$inner_rows) + 1L]] <<- list(
      sp = unname(as.numeric(sp)),
      scale = unname(as.numeric(scale)),
      iter = unname(as.integer(iter)),
      deviance = unname(as.numeric(dev)),
      pdev = unname(as.numeric(pdev)),
      old_pdev = unname(as.numeric(old.pdev))
    )
  }),
  exit = quote({
    value <- returnValue()
    if (identical(scoreType, "REML") && deriv == 1) {
      v1 <- -2 * weights * (y - value$fitted.values) *
        family$mu.eta(value$linear.predictors) / family$variance(value$fitted.values)
      dev_grad <- drop(crossprod(x, v1))
      raw_db_drho <- matrix(oo$b1, ncol(x), length(sp))
      dev_D1 <- drop(crossprod(raw_db_drho, dev_grad))
      v1_pre <- -2 * weg * (yg - mug) / (V * g1)
      dev_D1_pre <- drop(crossprod(raw_db_drho, drop(crossprod(x[good, ], v1_pre))))
      raw_beta <- unname(as.numeric(oo$beta))
      rS_matrix <- do.call(cbind, rS)
      root_work <- drop(crossprod(rS_matrix, raw_beta)) * exp(sp)
      Skb <- drop(rS_matrix %*% root_work)
      direct_bSb1 <- sum(raw_beta * Skb)
      Sb <- drop(crossprod(Sr, Sr %*% raw_beta))
      indirect_bSb1 <- 2 * drop(crossprod(raw_db_drho, Sb))
      probe_env$fits[[length(probe_env$fits) + 1L]] <<- list(
        sp = unname(as.numeric(sp)),
        scale = unname(as.numeric(scale)),
        iter = unname(as.integer(value$iter)),
        deviance = unname(as.numeric(value$deviance)),
        score = unname(as.numeric(value$REML)),
        grad = unname(as.numeric(value$REML1)),
        coef = unname(as.numeric(value$coefficients)),
        D1 = unname(as.numeric(oo$D1)),
        dev_D1 = unname(as.numeric(dev_D1)),
        dev_D1_pre = unname(as.numeric(dev_D1_pre)),
        bSb1 = unname(as.numeric(oo$D1 - dev_D1)),
        direct_bSb1 = unname(as.numeric(direct_bSb1)),
        indirect_bSb1 = unname(as.numeric(indirect_bSb1)),
        db_drho = unname(as.numeric(raw_db_drho)),
        beta_raw = unname(as.numeric(oo$beta)),
        E = unname(as.numeric(Sr)),
        rS = unname(as.numeric(do.call(cbind, rS))),
        dev_grad = unname(as.numeric(dev_grad)),
        trA1 = unname(as.numeric(oo$trA1)),
        ldetS1 = unname(as.numeric(rp$det1))
      )
    }
  })
)

bfgs_fun <- get("bfgs", envir = asNamespace("mgcv"))
trace(
  "bfgs",
  where = asNamespace("mgcv"),
  print = FALSE,
  at = list(deepest_path(bfgs_fun, "max.step <- 200")),
  tracer = quote({
    probe_env$initial <<- list(
      lsp = unname(as.numeric(lsp)),
      score = unname(as.numeric(score)),
      grad = unname(as.numeric(grad)),
      dVkk = unname(as.numeric(initial$dVkk)),
      fdgrad = unname(as.numeric(fdgrad))
    )
    probe_env$inverse_hessian <<- as.matrix(B)
  })
)

on.exit(try(untrace("bfgs", where = asNamespace("mgcv")), silent = TRUE), add = TRUE)
on.exit(try(untrace("gam.fit3", where = asNamespace("mgcv")), silent = TRUE), add = TRUE)

fit <- mgcv::gam(
  stats::as.formula(formula_text),
  data = data,
  family = Gamma(link = "inverse"),
  method = "REML",
  optimizer = c("outer", "bfgs")
)

print(probe_env$initial)
print(probe_env$fits[seq_len(min(4L, length(probe_env$fits)))])
print(probe_env$inner_rows[seq_len(min(8L, length(probe_env$inner_rows)))])
print(probe_env$inverse_hessian)
print(solve(probe_env$inverse_hessian))
print(fit$outer.info$score.hist)
