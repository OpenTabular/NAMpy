# Usage:
#   Rscript mgcv_general_family_preoptimization.R <csv_path> <output_json> <formula> <family> <method> <select> [sp_json]
#
# Fits an mgcv general family, re-creates the exact pre-optimization
# `Sl.setup` / `Sl.initial.repara` / `ldetS` state at the fitted smoothing
# parameters, and writes the setup objects as JSON.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop(
    "Usage: Rscript mgcv_general_family_preoptimization.R <csv_path> <output_json> <formula> <family> <method> <select> [sp_json]"
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

coerce_formula_list <- function(x) {
  if (is.character(x)) {
    lapply(as.list(x), as.formula)
  } else if (inherits(x, "formula")) {
    list(x)
  } else if (is.list(x)) {
    lapply(x, function(f) {
      if (inherits(f, "formula")) f else as.formula(f)
    })
  } else {
    stop("Unsupported formula specification.")
  }
}

serialize_optional <- function(x) {
  if (is.null(x)) {
    NULL
  } else {
    unname(x)
  }
}

serialize_offset_list <- function(offset) {
  if (is.null(offset)) {
    return(NULL)
  }
  if (!is.list(offset)) {
    return(list(unname(as.numeric(offset))))
  }
  lapply(offset, function(x) {
    if (is.null(x)) NULL else unname(as.numeric(x))
  })
}

serialize_sl_block <- function(block) {
  list(
    start = as.integer(block$start),
    stop = as.integer(block$stop),
    rank = if (is.null(block$rank)) NULL else as.integer(block$rank),
    ldet = if (is.null(block$ldet)) NULL else unname(as.numeric(block$ldet)),
    repara = isTRUE(block$repara),
    linear = if (is.null(block$linear)) TRUE else isTRUE(block$linear),
    lambda = unname(as.numeric(block$lambda)),
    ind = if (is.null(block$ind)) NULL else unname(as.logical(block$ind)),
    D = serialize_optional(block$D),
    Di = serialize_optional(block$Di),
    S = lapply(block$S, function(Si) unname(Si)),
    rS = if (is.null(block$rS)) list() else lapply(block$rS, function(M) unname(M)),
    St = serialize_optional(block$St)
  )
}

csv_path <- args[[1]]
output_json <- args[[2]]
formula_text <- normalize_formula_text(args[[3]])
family_name <- tolower(args[[4]])
method_name <- args[[5]]
select_flag <- tolower(args[[6]]) %in% c("true", "1", "yes")
sp_override <- if (length(args) >= 7 && nzchar(args[[7]]) && args[[7]] != "-") {
  as.numeric(jsonlite::fromJSON(args[[7]]))
} else {
  NULL
}

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

formula_raw <- NULL
if (grepl("^\\s*(c|list)\\s*\\(", formula_text)) {
  formula_raw <- eval(parse(text = formula_text))
}
formula_obj <- if (is.null(formula_raw)) as.formula(formula_text) else coerce_formula_list(formula_raw)

family_parts <- strsplit(family_name, ":", fixed = TRUE)[[1]]
family_key <- family_parts[[1]]
family_obj <- switch(
  family_key,
  gaulss = mgcv::gaulss(),
  gammals = mgcv::gammals(),
  stop(sprintf("Unsupported family for general-family pre-optimization parity: %s", family_name))
)

if (is.null(sp_override)) {
  fit <- gam(
    formula = formula_obj,
    data = data,
    family = family_obj,
    method = method_name,
    select = select_flag
  )
  fit_sp <- fit$sp
} else {
  fit <- NULL
  fit_sp <- sp_override
}

prefit <- gam(
  formula = formula_obj,
  data = data,
  family = family_obj,
  method = method_name,
  select = select_flag,
  sp = fit_sp,
  fit = FALSE
)

prefit$family <- mgcv:::fix.family(prefit$family)

X_full <- prefit$X
lpi <- attr(X_full, "lpi")
if (is.null(lpi)) {
  lpi <- list(seq_len(ncol(X_full)))
}

prefit$Sl <- mgcv:::Sl.setup(prefit)
X_initial <- mgcv:::Sl.initial.repara(
  prefit$Sl,
  X_full,
  inverse = FALSE,
  both.sides = FALSE,
  cov = FALSE
)

log_sp <- unname(as.numeric(log(pmax(fit_sp, 1e-300))))
ld <- mgcv:::ldetS(
  prefit$Sl,
  rho = log_sp,
  fixed = rep(FALSE, length(log_sp)),
  np = ncol(X_full),
  root = FALSE,
  Stot = FALSE,
  repara = TRUE
)

np <- ncol(X_full)
S_blocks <- vector("list", length(prefit$S))
St_full <- matrix(0, np, np)
if (length(prefit$S) > 0) {
  for (i in seq_along(prefit$S)) {
    Si_full <- matrix(0, np, np)
    ind <- prefit$off[i]:(prefit$off[i] + nrow(prefit$S[[i]]) - 1)
    Si_full[ind, ind] <- prefit$S[[i]]
    S_blocks[[i]] <- unname(Si_full)
    St_full <- St_full + fit_sp[i] * Si_full
  }
}

St_eig <- eigen((St_full + t(St_full)) / 2, symmetric = TRUE, only.values = TRUE)$values
St_tol <- max(max(St_eig), 0) * .Machine$double.eps^.75
Mp <- ncol(St_full) - sum(St_eig > St_tol)

payload <- list(
  X_full = unname(X_full),
  X_initial = unname(X_initial),
  jj = lapply(lpi, function(v) unname(as.integer(v) - 1L)),
  offset_list = serialize_offset_list(prefit$offset),
  smoothing_params = unname(as.numeric(fit_sp)),
  log_sp = log_sp,
  St = unname(St_full),
  S_blocks = S_blocks,
  ldetS = serialize_optional(ld$ldetS),
  ldetS1 = serialize_optional(ld$ldet1),
  ldetS2 = serialize_optional(ld$ldet2),
  Mp = as.integer(Mp),
  score_type = if (tolower(method_name) == "laml") "REML" else toupper(method_name),
  Sl = list(
    blocks = lapply(prefit$Sl, serialize_sl_block),
    E = serialize_optional(attr(prefit$Sl, "E")),
    S = serialize_optional(attr(prefit$Sl, "S")),
    lambda = serialize_optional(attr(prefit$Sl, "lambda")),
    cholesky = isTRUE(attr(prefit$Sl, "cholesky"))
  )
)

write_json(
  payload,
  output_json,
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
