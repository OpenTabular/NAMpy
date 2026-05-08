# Usage:
#   Rscript mgcv_fixed_sp_fit5_select_true.R <csv_path> <output_json> <formula> <family> <sp_json> [<score_type>]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop(
    "Usage: Rscript mgcv_fixed_sp_fit5_select_true.R <csv_path> <output_json> <formula> <family> <sp_json> [<score_type>]"
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
  if (is.null(x)) NULL else unname(x)
}

csv_path <- args[[1]]
output_json <- args[[2]]
formula_text <- normalize_formula_text(args[[3]])
family_name <- tolower(args[[4]])
sp <- as.numeric(jsonlite::fromJSON(args[[5]]))
score_type <- if (length(args) >= 6) toupper(args[[6]]) else "REML"

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
  stop(sprintf("Unsupported family for gam.fit5 fixed-sp parity: %s", family_name))
)

prefit <- gam(
  formula = formula_obj,
  data = data,
  family = family_obj,
  method = "REML",
  sp = unname(sp),
  fit = FALSE,
  select = TRUE
)

prefit$family <- mgcv:::fix.family(prefit$family)
prefit$Sl <- mgcv:::Sl.setup(prefit)
x_initial <- mgcv:::Sl.initial.repara(prefit$Sl, prefit$X, inverse = FALSE, both.sides = FALSE, cov = FALSE)
initial_sp <- if (length(prefit$S) > 0) {
  mgcv:::initial.spg(
    prefit$X,
    prefit$y,
    prefit$w,
    family_obj,
    prefit$S,
    prefit$rank,
    prefit$off,
    offset = prefit$offset,
    L = prefit$L,
    lsp0 = prefit$lsp0,
    E = as.matrix(prefit$Eb)
  )
} else {
  numeric(0)
}

np_total <- ncol(prefit$X)
St_full <- matrix(0, np_total, np_total)
if (length(prefit$S) > 0) {
  for (i in seq_along(prefit$S)) {
    ind <- prefit$off[i]:(prefit$off[i] + nrow(prefit$S[[i]]) - 1)
    St_full[ind, ind] <- St_full[ind, ind] + sp[i] * prefit$S[[i]]
  }
}
St_eig <- eigen((St_full + t(St_full)) / 2, symmetric = TRUE, only.values = TRUE)$values
St_tol <- max(max(St_eig), 0) * .Machine$double.eps^.75
Mp <- ncol(St_full) - sum(St_eig > St_tol)

fit <- mgcv:::gam.fit5(
  x = x_initial,
  y = prefit$y,
  lsp = log(pmax(unname(sp), 1e-300)),
  Sl = prefit$Sl,
  weights = prefit$w,
  offset = prefit$offset,
  deriv = 2,
  family = prefit$family,
  scoreType = score_type,
  control = gam.control(),
  Mp = Mp,
  gamma = 1
)

payload <- list(
  initial_sp = serialize_optional(initial_sp),
  coefficients_full = serialize_optional(
    mgcv:::Sl.initial.repara(
      prefit$Sl,
      fit$coefficients,
      inverse = TRUE,
      both.sides = FALSE,
      cov = FALSE
    )
  ),
  REML = unname(as.numeric(fit$REML)),
  REML1 = serialize_optional(fit$REML1),
  REML2 = serialize_optional(fit$REML2),
  db_drho = serialize_optional(fit$db.drho),
  db_drho_full = serialize_optional(
    if (is.null(fit$db.drho)) {
      NULL
    } else {
      sapply(seq_len(ncol(fit$db.drho)), function(i) {
        mgcv:::Sl.initial.repara(
          prefit$Sl,
          fit$db.drho[, i],
          inverse = TRUE,
          both.sides = FALSE,
          cov = FALSE
        )
      })
    }
  ),
  outer_info = list(
    hess = if (!is.null(fit$outer.info) && !is.null(fit$outer.info$hess)) unname(as.matrix(fit$outer.info$hess)) else NULL,
    grad = if (!is.null(fit$outer.info) && !is.null(fit$outer.info$grad)) unname(as.numeric(fit$outer.info$grad)) else NULL
  )
)

write_json(payload, output_json, auto_unbox = TRUE, digits = 17, pretty = TRUE, null = "null")
