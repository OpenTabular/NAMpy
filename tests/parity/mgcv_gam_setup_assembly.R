# Usage:
#   Rscript mgcv_gam_setup_assembly.R <csv_path> <output_json> <formula> <family> <method> <select>
#
# Dump mgcv::gam(..., fit = FALSE) assembly payload:
#   G$X, G$S, G$off, G$rank, G$L, G$lsp0, G$sp, G$smooth,
#   G$P, G$cmX, G$assign, G$xlevels, G$offset, G$y.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop(
    "Usage: Rscript mgcv_gam_setup_assembly.R <csv_path> <output_json> <formula> <family> <method> <select>"
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

as_char_or_null <- function(x) {
  if (is.null(x)) return(NULL)
  out <- as.character(x)
  if (length(out) == 1L && (is.na(out) || identical(out, "NA"))) return(NULL)
  unname(out)
}

serialize_integer_or_list <- function(x) {
  if (is.null(x)) return(NULL)
  if (is.list(x)) {
    return(lapply(x, function(v) {
      if (is.null(v)) NULL else unname(as.integer(v))
    }))
  }
  unname(as.integer(x))
}

serialize_xlevels <- function(x) {
  if (is.null(x)) return(list())
  if (is.list(x) && !is.null(names(x)) && any(nzchar(names(x)))) {
    return(lapply(x, function(v) {
      if (is.null(v)) NULL else unname(as.character(v))
    }))
  }
  if (is.list(x)) {
    return(lapply(x, function(v) {
      if (is.null(v)) {
        NULL
      } else if (is.list(v)) {
        lapply(v, function(u) {
          if (is.null(u)) NULL else unname(as.character(u))
        })
      } else {
        unname(as.character(v))
      }
    }))
  }
  list()
}

serialize_offset <- function(x) {
  if (is.null(x)) return(NULL)
  if (is.list(x)) {
    return(lapply(x, function(v) {
      if (is.null(v)) NULL else unname(as.numeric(v))
    }))
  }
  unname(as.numeric(x))
}

smooth_special <- function(sm) {
  if (inherits(sm, "tensor.smooth")) {
    if (isTRUE(sm[["inter", exact = TRUE]])) return("ti")
    return("te")
  }
  "s"
}

smooth_basis <- function(sm) {
  margin <- sm[["margin", exact = TRUE]]
  if (!is.null(margin)) {
    bs_vals <- vapply(
      margin,
      function(m) {
        bs <- m[["bs", exact = TRUE]]
        if (is.null(bs)) return(NA_character_)
        as.character(bs[[1]])
      },
      character(1)
    )
    if (!all(is.na(bs_vals))) return(unname(bs_vals[!is.na(bs_vals)]))
    return(
      NULL
    )
  }
  bs <- sm[["bs", exact = TRUE]]
  if (!is.null(bs)) return(as.character(bs[[1]]))
  NULL
}

serialize_smooth_setup <- function(sm) {
  list(
    class_name = as.character(class(sm)[1]),
    special = smooth_special(sm),
    basis = smooth_basis(sm),
    label = as_char_or_null(sm[["label", exact = TRUE]]),
    term = if (is.null(sm[["term", exact = TRUE]])) NULL else unname(as.character(sm[["term", exact = TRUE]])),
    by_name = as_char_or_null(sm[["by", exact = TRUE]]),
    by_level = as_char_or_null(sm[["by.level", exact = TRUE]]),
    id = as_char_or_null(sm[["id", exact = TRUE]]),
    dim = if (is.null(sm[["dim", exact = TRUE]])) NULL else as.integer(sm[["dim", exact = TRUE]]),
    df = if (is.null(sm[["df", exact = TRUE]])) NULL else as.integer(sm[["df", exact = TRUE]]),
    del_index = {
      del <- attr(sm, "del.index")
      if (is.null(del)) NULL else unname(as.integer(del))
    },
    side_constrain = if (is.null(sm[["side.constrain", exact = TRUE]])) NULL else isTRUE(sm[["side.constrain", exact = TRUE]]),
    first_para = if (is.null(sm[["first.para", exact = TRUE]])) NULL else as.integer(sm[["first.para", exact = TRUE]]),
    last_para = if (is.null(sm[["last.para", exact = TRUE]])) NULL else as.integer(sm[["last.para", exact = TRUE]]),
    first_sp = if (is.null(sm[["first.sp", exact = TRUE]])) NULL else as.integer(sm[["first.sp", exact = TRUE]]),
    last_sp = if (is.null(sm[["last.sp", exact = TRUE]])) NULL else as.integer(sm[["last.sp", exact = TRUE]]),
    sp = if (is.null(sm[["sp", exact = TRUE]])) NULL else unname(as.numeric(sm[["sp", exact = TRUE]])),
    rank = if (is.null(sm[["rank", exact = TRUE]])) NULL else unname(as.integer(sm[["rank", exact = TRUE]])),
    null_space_dim = if (is.null(sm[["null.space.dim", exact = TRUE]])) NULL else as.integer(sm[["null.space.dim", exact = TRUE]]),
    n_penalties = length(sm[["S", exact = TRUE]]),
    n_coef = if (
      is.null(sm[["first.para", exact = TRUE]]) || is.null(sm[["last.para", exact = TRUE]])
    ) NULL else as.integer(sm[["last.para", exact = TRUE]] - sm[["first.para", exact = TRUE]] + 1L),
    full = if (!is.null(sm[["full", exact = TRUE]])) isTRUE(sm[["full", exact = TRUE]]) else NULL,
    ord = if (is.null(sm[["ord", exact = TRUE]])) NULL else unname(as.integer(sm[["ord", exact = TRUE]]))
  )
}

csv_path <- args[[1]]
output_json <- args[[2]]
formula_text <- normalize_formula_text(args[[3]])
family_name <- tolower(args[[4]])
method_name <- toupper(args[[5]])
select_flag <- tolower(args[[6]]) == "true"

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
family_param <- if (length(family_parts) >= 2) family_parts[[2]] else NULL

family_obj <- switch(
  family_key,
  gaussian = gaussian(),
  binomial = binomial(link = if (is.null(family_param)) "logit" else family_param),
  poisson = poisson(link = if (is.null(family_param)) "log" else family_param),
  gamma = Gamma(link = if (is.null(family_param)) "inverse" else family_param),
  negbin_est = {
    theta <- if (is.null(family_param)) 1.0 else as.numeric(family_param)
    mgcv::nb(theta = -abs(theta), link = "log")
  },
  negbin = {
    theta <- if (is.null(family_param)) 1.0 else as.numeric(family_param)
    mgcv::nb(theta = theta, link = "log")
  },
  gaulss = mgcv::gaulss(),
  gammals = mgcv::gammals(),
  stop(sprintf("Unsupported family token for gam.setup assembly parity: %s", family_name))
)

fit_method <- if (tolower(method_name) == "fixed") "REML" else method_name

prefit <- gam(
  formula = formula_obj,
  data = data,
  family = family_obj,
  method = fit_method,
  fit = FALSE,
  select = select_flag
)

fit_obj <- gam(
  formula = formula_obj,
  data = data,
  family = family_obj,
  method = fit_method,
  select = select_flag
)

Xp <- tryCatch(
  unname(as.matrix(predict(fit_obj, type = "lpmatrix"))),
  error = function(e) NULL
)

payload <- list(
  X = unname(prefit$X),
  Xp = Xp,
  S = lapply(prefit$S, function(Si) unname(Si)),
  off = unname(as.integer(prefit$off)),
  rank = unname(as.integer(prefit$rank)),
  L = if (is.null(prefit$L)) NULL else unname(prefit$L),
  lsp0 = if (is.null(prefit$lsp0)) NULL else unname(as.numeric(prefit$lsp0)),
  sp = if (is.null(prefit$sp)) NULL else unname(as.numeric(prefit$sp)),
  smooth = lapply(prefit$smooth, serialize_smooth_setup),
  P = if (is.null(prefit$P)) NULL else unname(prefit$P),
  cmX = if (is.null(prefit$cmX)) NULL else unname(as.numeric(prefit$cmX)),
  assign = serialize_integer_or_list(prefit$assign),
  xlevels = serialize_xlevels(prefit$xlevels),
  offset = serialize_offset(prefit$offset),
  y = if (is.null(prefit$y)) NULL else unname(as.numeric(prefit$y))
)

write_json(payload, output_json, auto_unbox = TRUE, digits = 17, pretty = TRUE, null = "null")
