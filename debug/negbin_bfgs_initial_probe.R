# Usage:
#   Rscript debug/negbin_bfgs_initial_probe.R <csv_path> <formula>
#
# Captures mgcv:::bfgs()'s initial joint (log theta, log sp) state.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: negbin_bfgs_initial_probe.R <csv_path> <formula>")

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

data <- read.csv(args[[1]], stringsAsFactors = FALSE)
formula <- stats::as.formula(args[[2]])
probe_env <- new.env(parent = emptyenv())
probe_env$initial <- NULL
probe_env$inverse_hessian <- NULL

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

fit <- mgcv::gam(
  formula,
  data = data,
  family = mgcv::nb(theta = -1.8),
  method = "REML",
  optimizer = c("outer", "bfgs")
)

print(probe_env$initial)
print(probe_env$inverse_hessian)
print(solve(probe_env$inverse_hessian))
print(fit$outer.info$score.hist)
