# Usage:
#   Rscript mgcv_outer_trace.R <csv_path> <output_json> <formula> <family> <method> <optimizer> <select> <edge_correct>
#
# Captures mgcv outer smoothing-optimization traces for `newton`, `bfgs`,
# `optim`, and `efs` via lightweight instrumentation of the upstream mgcv
# namespace. The payload is normalized so Python parity tests can compare one
# schema across optimizers.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 8) {
  stop(
    paste(
      "Usage: Rscript mgcv_outer_trace.R <csv_path> <output_json>",
      "<formula> <family> <method> <optimizer> <select> <edge_correct>"
    )
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

method_token <- function(x) {
  key <- tolower(x)
  if (key %in% c("gcv", "gcv.cp")) return("GCV.Cp")
  if (key == "ubre") return("UBRE")
  if (key == "ml") return("ML")
  if (key == "reml") return("REML")
  x
}

family_object <- function(family_name) {
  family_parts <- strsplit(tolower(family_name), ":", fixed = TRUE)[[1]]
  family_key <- family_parts[[1]]
  family_param <- if (length(family_parts) >= 2) family_parts[[2]] else NULL
  switch(
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
      theta <- if (is.null(family_param) || family_param == "") 1.0 else as.numeric(family_param)
      mgcv::nb(theta = theta, link = "log")
    },
    negbin_est = {
      theta <- if (is.null(family_param) || family_param == "") 1.0 else as.numeric(family_param)
      mgcv::nb(theta = -abs(theta), link = "log")
    },
    gaulss = mgcv::gaulss(),
    gammals = mgcv::gammals(),
    ziplss = mgcv::ziplss(),
    gevlss = mgcv::gevlss(),
    shash = mgcv::shash(),
    shashlss = mgcv::shash(),
    stop(sprintf("Unsupported family for outer trace parity: %s", family_name))
  )
}

coerce_formula <- function(x) {
  obj <- eval(parse(text = x))
  if (is.character(obj)) {
    if (length(obj) == 1) return(as.formula(obj))
    return(lapply(obj, as.formula))
  }
  obj
}

find_paths <- function(expr, pattern, path = integer()) {
  hits <- list()
  txt <- paste(deparse(expr, width.cutoff = 500L), collapse = " ")
  if (grepl(pattern, txt, fixed = TRUE)) {
    hits[[length(hits) + 1L]] <- path
  }
  if (is.call(expr) || is.pairlist(expr) || is.expression(expr)) {
    parts <- as.list(expr)
    for (i in seq_along(parts)) {
      hits <- c(hits, find_paths(parts[[i]], pattern, c(path, i)))
    }
  }
  hits
}

deepest_path <- function(fun, pattern) {
  hits <- Filter(length, find_paths(body(fun), pattern))
  if (length(hits) == 0L) {
    stop(sprintf("Failed to locate pattern in mgcv:::%s: %s", deparse(substitute(fun)), pattern))
  }
  hits[[which.max(vapply(hits, length, integer(1)))]]
}

trace_env <- new.env(parent = emptyenv())
trace_env$trace_rows <- list()
trace_env$optim_rows <- new.env(hash = TRUE, parent = emptyenv())
trace_env$optim_order <- character()

append_trace_row <- function(row) {
  prev <- if (length(trace_env$trace_rows) == 0L) NULL else trace_env$trace_rows[[length(trace_env$trace_rows)]]
  lsp_full <- as.numeric(row$log_sp_full)
  step_norm <- if (is.null(prev)) 0.0 else sqrt(sum((lsp_full - as.numeric(prev$log_sp_full))^2))
  row$accepted_step_norm <- as.numeric(step_norm)
  trace_env$trace_rows[[length(trace_env$trace_rows) + 1L]] <- row
}

optim_key <- function(x) {
  paste(format(as.numeric(x), digits = 17, scientific = TRUE, trim = TRUE), collapse = "|")
}

record_optim_eval <- function(kind, lsp, value) {
  key <- optim_key(lsp)
  if (!exists(key, envir = trace_env$optim_rows, inherits = FALSE)) {
    assign(
      key,
      list(
        log_sp_full = as.numeric(lsp),
        criterion = NULL,
        gradient = NULL,
        hessian = NULL,
        n_fun = 0L,
        n_jac = 0L
      ),
      envir = trace_env$optim_rows
    )
    trace_env$optim_order <- c(trace_env$optim_order, key)
  }
  row <- get(key, envir = trace_env$optim_rows, inherits = FALSE)
  if (identical(kind, "fun")) {
    row$criterion <- as.numeric(value)
    row$n_fun <- as.integer(row$n_fun) + 1L
  } else {
    row$gradient <- as.numeric(value)
    row$n_jac <- as.integer(row$n_jac) + 1L
  }
  assign(key, row, envir = trace_env$optim_rows)
}

wrap_ns_fun <- function(name, wrapper_builder) {
  ns <- asNamespace("mgcv")
  orig <- get(name, envir = ns)
  unlockBinding(name, ns)
  assign(name, wrapper_builder(orig), envir = ns)
  lockBinding(name, ns)
  orig
}

restore_ns_fun <- function(name, orig) {
  ns <- asNamespace("mgcv")
  unlockBinding(name, ns)
  assign(name, orig, envir = ns)
  lockBinding(name, ns)
}

active_traces <- list()
wrapped_funs <- list()

register_trace <- function(name, path, tracer_expr) {
  trace(name, where = asNamespace("mgcv"), print = FALSE, at = list(path), tracer = tracer_expr)
  active_traces[[length(active_traces) + 1L]] <<- name
}

cleanup <- function() {
  for (nm in rev(active_traces)) {
    try(untrace(nm, where = asNamespace("mgcv")), silent = TRUE)
  }
  for (nm in names(wrapped_funs)) {
    try(restore_ns_fun(nm, wrapped_funs[[nm]]), silent = TRUE)
  }
}

on.exit(cleanup(), add = TRUE)

suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))

csv_path <- args[[1]]
output_json <- args[[2]]
formula_text <- normalize_formula_text(args[[3]])
family_name <- args[[4]]
method_name <- method_token(args[[5]])
optimizer_name <- tolower(args[[6]])
select_flag <- tolower(args[[7]]) %in% c("true", "1", "yes")
edge_correct_flag <- tolower(args[[8]]) %in% c("true", "1", "yes")
family_key <- strsplit(tolower(family_name), ":", fixed = TRUE)[[1]][1]

data <- read.csv(csv_path, stringsAsFactors = FALSE)
for (nm in names(data)) {
  if (is.character(data[[nm]])) data[[nm]] <- factor(data[[nm]])
}

family_obj <- family_object(family_name)

if (optimizer_name == "newton") {
  newton_fun <- get("newton", envir = asNamespace("mgcv"))
  register_trace(
    "newton",
    deepest_path(newton_fun, "if (converged) break"),
    quote({
      append_trace_row(
        list(
          iter = as.integer(i),
          log_sp_full = unname(as.numeric(lsp)),
          criterion = unname(as.numeric(score)),
          gradient = unname(as.numeric(grad)),
          hessian = unname(as.matrix(hess)),
          step_halving_count = as.integer(ii),
          rank_info = list(
            source = "mgcv_newton",
            indefinite_hessian = as.logical(indef),
            positive_definite = as.logical(pdef),
            step_halving_count = as.integer(ii),
            converged_here = as.logical(converged)
          )
        )
      )
    })
  )
} else if (optimizer_name == "bfgs") {
  bfgs_fun <- get("bfgs", envir = asNamespace("mgcv"))
  register_trace(
    "bfgs",
    deepest_path(bfgs_fun, "initial <- trial"),
    quote({
      append_trace_row(
        list(
          iter = as.integer(i),
          log_sp_full = unname(as.numeric(ilsp)),
          criterion = unname(as.numeric(trial$score)),
          gradient = unname(as.numeric(trial$grad)),
          hessian = NULL,
          line_search_alpha = unname(as.numeric(trial$alpha)),
          rank_info = list(
            source = "mgcv_bfgs",
            line_search_alpha = unname(as.numeric(trial$alpha)),
            converged_here = as.logical(converged),
            rolled_back = as.logical(rolled.back)
          )
        )
      )
    })
  )
} else if (optimizer_name == "efs") {
  efs_name <- if (inherits(family_obj, "general.family")) "efsud" else "efsudr"
  efs_fun <- get(efs_name, envir = asNamespace("mgcv"))
  register_trace(
    efs_name,
    deepest_path(efs_fun, "score.hist[iter] <- fit$REML"),
    quote({
      append_trace_row(
        list(
          iter = as.integer(iter),
          log_sp_full = unname(as.numeric(lsp)),
          criterion = unname(as.numeric(fit$REML)),
          gradient = NULL,
          hessian = NULL,
          max_step = unname(as.numeric(max.step)),
          rank_info = list(
            source = "mgcv_efs",
            mult = unname(as.numeric(mult)),
            max_step = unname(as.numeric(max.step))
          )
        )
      )
    })
  )
} else if (optimizer_name == "optim") {
  wrapped_funs[["gam2objective"]] <- wrap_ns_fun("gam2objective", function(orig) {
    function(lsp, args, ...) {
      val <- orig(lsp, args, ...)
      record_optim_eval("fun", lsp, val)
      val
    }
  })
  wrapped_funs[["gam2derivative"]] <- wrap_ns_fun("gam2derivative", function(orig) {
    function(lsp, args, ...) {
      val <- orig(lsp, args, ...)
      record_optim_eval("grad", lsp, val)
      val
    }
  })
} else {
  stop(sprintf("Unsupported optimizer for outer trace parity: %s", optimizer_name))
}

optimizer_arg <- if (optimizer_name == "efs") {
  "efs"
} else {
  c("outer", optimizer_name)
}

control_arg <- gam.control(edge.correct = edge_correct_flag)

fit <- gam(
  formula = coerce_formula(formula_text),
  data = data,
  family = family_obj,
  method = method_name,
  optimizer = optimizer_arg,
  select = select_flag,
  control = control_arg
)

split_scale_blocks <- function(log_sp_full, gradient, hessian, n_sp, extra_kind = NULL) {
  log_sp_full <- as.numeric(log_sp_full)
  gradient <- if (is.null(gradient)) NULL else as.numeric(gradient)
  hessian <- if (is.null(hessian)) NULL else as.matrix(hessian)

  out <- list(
    log_sp = log_sp_full,
    log_scale = NULL,
    log_theta = NULL,
    gradient = gradient,
    gradient_full = gradient,
    hessian = hessian,
    hessian_full = hessian
  )

  if (length(log_sp_full) > n_sp) {
    extra_val <- unname(as.numeric(log_sp_full[length(log_sp_full)]))
    out$log_sp <- log_sp_full[seq_len(n_sp)]
    if (identical(extra_kind, "theta")) {
      out$log_theta <- extra_val
    } else {
      out$log_scale <- extra_val
    }
  }
  if (!is.null(gradient) && length(gradient) > n_sp) {
    out$gradient <- gradient[seq_len(n_sp)]
  }
  if (!is.null(hessian) && nrow(hessian) > n_sp) {
    out$hessian <- hessian[seq_len(n_sp), seq_len(n_sp), drop = FALSE]
  }
  out
}

n_sp <- length(fit$sp)
extra_kind <- if (family_key == "negbin_est") "theta" else if (family_key == "gamma") "scale" else NULL

trace_rows <- list()
if (optimizer_name == "optim") {
  prev_lsp <- NULL
  for (i in seq_along(trace_env$optim_order)) {
    key <- trace_env$optim_order[[i]]
    row <- get(key, envir = trace_env$optim_rows, inherits = FALSE)
    split <- split_scale_blocks(
      row$log_sp_full,
      row$gradient,
      row$hessian,
      n_sp = n_sp,
      extra_kind = extra_kind
    )
    step_norm <- if (is.null(prev_lsp)) 0.0 else sqrt(sum((as.numeric(split$log_sp) - prev_lsp)^2))
    trace_rows[[length(trace_rows) + 1L]] <- list(
      iter = as.integer(i - 1L),
      log_sp = unname(as.numeric(split$log_sp)),
      log_scale = split$log_scale,
      log_theta = split$log_theta,
      criterion = if (is.null(row$criterion)) NULL else unname(as.numeric(row$criterion)),
      gradient = if (is.null(split$gradient)) NULL else unname(as.numeric(split$gradient)),
      gradient_full = if (is.null(split$gradient_full)) NULL else unname(as.numeric(split$gradient_full)),
      hessian = NULL,
      hessian_full = NULL,
      accepted_step_norm = unname(as.numeric(step_norm)),
      rank_info = list(
        source = "mgcv_optim",
        n_fun = as.integer(row$n_fun),
        n_jac = as.integer(row$n_jac)
      )
    )
    prev_lsp <- as.numeric(split$log_sp)
  }
} else {
  for (row in trace_env$trace_rows) {
    split <- split_scale_blocks(
      row$log_sp_full,
      row$gradient,
      row$hessian,
      n_sp = n_sp,
      extra_kind = extra_kind
    )
    trace_rows[[length(trace_rows) + 1L]] <- list(
      iter = as.integer(row$iter),
      log_sp = unname(as.numeric(split$log_sp)),
      log_scale = split$log_scale,
      log_theta = split$log_theta,
      criterion = if (is.null(row$criterion)) NULL else unname(as.numeric(row$criterion)),
      gradient = if (is.null(split$gradient)) NULL else unname(as.numeric(split$gradient)),
      gradient_full = if (is.null(split$gradient_full)) NULL else unname(as.numeric(split$gradient_full)),
      hessian = if (is.null(split$hessian)) NULL else unname(as.matrix(split$hessian)),
      hessian_full = if (is.null(split$hessian_full)) NULL else unname(as.matrix(split$hessian_full)),
      accepted_step_norm = unname(as.numeric(row$accepted_step_norm)),
      rank_info = row$rank_info
    )
  }
}

outer_info <- fit$outer.info
outer_grad <- if (!is.null(outer_info) && !is.null(outer_info$grad)) unname(as.numeric(outer_info$grad)) else NULL
outer_hess <- if (!is.null(outer_info) && !is.null(outer_info$hess)) unname(as.matrix(outer_info$hess)) else NULL
outer_split <- split_scale_blocks(
  if (n_sp > 0) log(fit$sp) else numeric(0),
  outer_grad,
  outer_hess,
  n_sp = n_sp,
  extra_kind = extra_kind
)

payload <- list(
  fit = list(
    criterion_name = method_name,
    smoothing_params = unname(as.numeric(fit$sp)),
    optimizer = if (is.null(fit$optimizer)) NULL else as.character(fit$optimizer),
    outer_info = list(
      optimizer = optimizer_name,
      conv = if (!is.null(outer_info) && !is.null(outer_info$conv)) as.character(outer_info$conv) else NULL,
      iter = if (!is.null(outer_info) && !is.null(outer_info$iter)) as.integer(outer_info$iter) else NULL,
      score_hist = if (!is.null(outer_info) && !is.null(outer_info$score.hist)) unname(as.numeric(outer_info$score.hist)) else NULL,
      log_scale = outer_split$log_scale,
      log_theta = outer_split$log_theta,
      gradient = outer_split$gradient,
      gradient_full = outer_split$gradient_full,
      hessian = if (is.null(outer_split$hessian)) NULL else unname(as.matrix(outer_split$hessian)),
      hessian_full = if (is.null(outer_split$hessian_full)) NULL else unname(as.matrix(outer_split$hessian_full)),
      edge_correct = !is.null(outer_hess) && !is.null(attr(outer_info$hess, "edge.correct")),
      lsp1 = if (!is.null(outer_info) && !is.null(outer_info$hess) && !is.null(attr(outer_info$hess, "lsp1"))) unname(as.numeric(attr(outer_info$hess, "lsp1")))[seq_len(min(n_sp, length(attr(outer_info$hess, "lsp1"))))] else NULL,
      hess1 = if (!is.null(outer_info) && !is.null(outer_info$hess) && !is.null(attr(outer_info$hess, "hess1"))) {
        h1 <- as.matrix(attr(outer_info$hess, "hess1"))
        if (nrow(h1) > n_sp) h1 <- h1[seq_len(n_sp), seq_len(n_sp), drop = FALSE]
        unname(h1)
      } else NULL,
      convergence = if (!is.null(outer_info) && !is.null(outer_info$convergence)) as.integer(outer_info$convergence) else NULL,
      message = if (!is.null(outer_info) && !is.null(outer_info$message)) as.character(outer_info$message) else NULL,
      counts = if (!is.null(outer_info) && !is.null(outer_info$counts)) unname(as.integer(outer_info$counts)) else NULL
    )
  ),
  trace = trace_rows
)

write_json(payload, output_json, auto_unbox = TRUE, digits = 17, pretty = TRUE)
