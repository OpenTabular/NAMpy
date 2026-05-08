# Usage:
#   Rscript mgcv_interpret_gam.R <input_json> <output_json>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript mgcv_interpret_gam.R <input_json> <output_json>")
}

normalize_formula_text <- function(x) {
  x <- gsub("\\*\\*", "^", x)
  x <- gsub("\\[", "c(", x)
  x <- gsub("\\]", ")", x)
  x <- gsub("\\bTrue\\b", "TRUE", x)
  x <- gsub("\\bFalse\\b", "FALSE", x)
  x <- gsub("\\bNone\\b", "NULL", x)
  x
}

kind_from_spec <- function(spec) {
  if (inherits(spec, "tensor.smooth.spec")) {
    if (isTRUE(spec$inter)) return("ti")
    return("te")
  }
  "s"
}

serialize_smooth <- function(spec) {
  k_val <- NULL
  if (inherits(spec, "tensor.smooth.spec")) {
    k_val <- lapply(spec$margin, function(m) unname(as.numeric(m$bs.dim)))
    k_val <- unname(as.numeric(unlist(k_val)))
  } else if (!is.null(spec$bs.dim)) {
    k_val <- list(unname(as.numeric(spec$bs.dim)))
  }

  list(
    kind = kind_from_spec(spec),
    label = as.character(spec$label),
    term = as.list(unname(as.character(spec$term))),
    by = if (is.null(spec$by)) NULL else as.character(spec$by),
    id = if (is.null(spec$id)) NULL else as.character(spec$id),
    k = if (is.null(k_val)) list() else as.list(k_val)
  )
}

serialize_component <- function(comp) {
  pterms <- terms.formula(comp$pf)
  pterm_labels <- attr(pterms, "term.labels")

  list(
    pf = paste(deparse(comp$pf, width.cutoff = 500L), collapse = " "),
    pfok = unname(as.integer(comp$pfok)),
    fake_formula = paste(
      deparse(comp$fake.formula, width.cutoff = 500L),
      collapse = " "
    ),
    response = if (is.null(comp$response)) NULL else as.character(comp$response),
    fake_names = as.list(unname(as.character(comp$fake.names))),
    pred_names = as.list(unname(as.character(comp$pred.names))),
    pred_formula = paste(
      deparse(comp$pred.formula, width.cutoff = 500L),
      collapse = " "
    ),
    lpi = if (is.null(comp$lpi)) list() else as.list(unname(as.integer(comp$lpi))),
    intercept = as.logical(attr(pterms, "intercept") > 0),
    parametric_terms = as.list(unname(as.character(pterm_labels))),
    smooth_terms = lapply(comp$smooth.spec, serialize_smooth)
  )
}

coerce_formula_spec <- function(x) {
  if (is.character(x) && length(x) == 1) {
    return(as.formula(normalize_formula_text(x)))
  }

  if (is.character(x)) {
    return(lapply(as.list(x), function(f) as.formula(normalize_formula_text(f))))
  }

  stop("Unsupported formula specification for interpret.gam parity.")
}

input_json <- args[[1]]
output_json <- args[[2]]

suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(mgcv))

spec <- fromJSON(input_json, simplifyVector = TRUE)
formula_obj <- coerce_formula_spec(spec$formula)
parsed <- mgcv::interpret.gam(formula_obj)

if (is.list(formula_obj) && !inherits(formula_obj, "formula")) {
  n_components <- length(formula_obj)
  components <- lapply(seq_len(n_components), function(i) serialize_component(parsed[[i]]))
  result <- list(
    response = if (is.null(parsed$response)) NULL else as.character(parsed$response),
    fake_formula = paste(
      deparse(parsed$fake.formula, width.cutoff = 500L),
      collapse = " "
    ),
    pred_formula = paste(
      deparse(parsed$pred.formula, width.cutoff = 500L),
      collapse = " "
    ),
    nlp = unname(as.integer(parsed$nlp)),
    components = components
  )
} else {
  result <- list(
    response = if (is.null(parsed$response)) NULL else as.character(parsed$response),
    fake_formula = paste(
      deparse(parsed$fake.formula, width.cutoff = 500L),
      collapse = " "
    ),
    pred_formula = paste(
      deparse(parsed$pred.formula, width.cutoff = 500L),
      collapse = " "
    ),
    nlp = 1L,
    components = list(serialize_component(parsed))
  )
}

write_json(result, output_json, auto_unbox = TRUE, pretty = TRUE, null = "null")
