# Usage:
#   Rscript mgcv_fixed_sp_fit5.R <csv_path> <output_json> <formula> <family> <sp_json> [<score_type>]
#
# Recreates mgcv's exact `Sl.setup` / `Sl.initial.repara` state and calls the
# low-level `gam.fit5()` fixed-sp inner optimizer.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop(
    "Usage: Rscript mgcv_fixed_sp_fit5.R <csv_path> <output_json> <formula> <family> <sp_json> [<score_type>]"
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

csv_path <- args[[1]]
output_json <- args[[2]]
formula_text <- normalize_formula_text(args[[3]])
family_name <- tolower(args[[4]])
sp <- as.numeric(jsonlite::fromJSON(args[[5]]))
score_type <- if (length(args) >= 6) toupper(args[[6]]) else "REML"
if (!(score_type %in% c("REML", "ML"))) {
  stop(sprintf("Unsupported gam.fit5 score type: %s", score_type))
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
  ziplss = mgcv::ziplss(),
  gevlss = mgcv::gevlss(),
  shash = mgcv::shash(),
  shashlss = mgcv::shash(),
  stop(sprintf("Unsupported family for gam.fit5 fixed-sp parity: %s", family_name))
)

prefit <- gam(
  formula = formula_obj,
  data = data,
  family = family_obj,
  method = "REML",
  sp = unname(sp),
  fit = FALSE
)

prefit$family <- mgcv:::fix.family(prefit$family)
prefit$Sl <- mgcv:::Sl.setup(prefit)
x_initial <- mgcv:::Sl.initial.repara(
  prefit$Sl,
  prefit$X,
  inverse = FALSE,
  both.sides = FALSE,
  cov = FALSE
)

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

rp_init <- mgcv:::ldetS(
  prefit$Sl,
  rho = log(pmax(unname(sp), 1e-300)),
  fixed = rep(FALSE, length(sp)),
  np = ncol(x_initial),
  root = TRUE,
  Stot = TRUE
)
x_fit <- mgcv:::Sl.repara(rp_init$rp, x_initial)
E_fit <- rp_init$E
attr(E_fit, "use.unscaled") <- TRUE
E <- E_fit
start <- NULL
start0 <- start
x <- x_fit
y <- prefit$y
weights <- prefit$w
offset <- prefit$offset
family <- prefit$family
nobs <- length(y)
eval(family$initialize)
start_initial <- start
if (!is.null(start0)) start <- start0

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

## mgcv:::gam.fit5() always returns the score in REML/REML1/REML2 slots,
## even when scoreType="ML".
score_val <- fit$REML
score1_val <- fit$REML1
score2_val <- fit$REML2

payload <- list(
  coefficients = unname(as.numeric(fit$coefficients)),
  coefficients_full = serialize_optional(
    mgcv:::Sl.initial.repara(
      prefit$Sl,
      fit$coefficients,
      inverse = TRUE,
      both.sides = FALSE,
      cov = FALSE
    )
  ),
  linear_predictors = serialize_optional(fit$linear.predictors),
  fitted_values = serialize_optional(fit$fitted.values),
  deviance = unname(as.numeric(-2 * fit$l)),
  loglik = unname(as.numeric(fit$l)),
  ldetHp = if (is.null(fit$L) || is.null(fit$D)) NULL else unname(
    as.numeric(2 * sum(log(diag(fit$L))) - 2 * sum(log(fit$D)))
  ),
  penalty_quadratic = if (is.null(fit$St)) NULL else unname(
    as.numeric(drop(t(fit$coefficients) %*% fit$St %*% fit$coefficients) / 2)
  ),
  score_type = score_type,
  score = unname(as.numeric(score_val)),
  score1 = serialize_optional(score1_val),
  score2 = serialize_optional(score2_val),
  REML = unname(as.numeric(score_val)),
  REML1 = serialize_optional(score1_val),
  REML2 = serialize_optional(score2_val),
  lbb = serialize_optional(fit$lbb),
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
  rank = unname(as.integer(fit$rank)),
  iter = unname(as.integer(fit$iter)),
  offset_list = serialize_offset_list(prefit$offset)
  ,
  start_initial = serialize_optional(start_initial)
)

write_json(
  payload,
  output_json,
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
