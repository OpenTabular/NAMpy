# Usage:
#   Rscript mgcv_anova.R <csv_path> <output_json> <formulas_json> <family> <method> <select> <test>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 7) {
  stop("Usage: Rscript mgcv_anova.R <csv_path> <output_json> <formulas_json> <family> <method> <select> <test>")
}

normalize_formula_text <- function(x) {
  x <- gsub("\\[", "c(", x)
  x <- gsub("\\]", ")", x)
  x <- gsub("\\bTrue\\b", "TRUE", x)
  x <- gsub("\\bFalse\\b", "FALSE", x)
  x <- gsub("\\bNone\\b", "NULL", x)
  x
}

coerce_formula_object <- function(txt) {
  formula_raw <- NULL
  if (grepl("^\\s*(c|list)\\s*\\(", txt)) {
    formula_raw <- eval(parse(text = txt))
  }
  if (is.null(formula_raw)) {
    return(as.formula(txt))
  }
  if (is.character(formula_raw)) {
    return(lapply(as.list(formula_raw), as.formula))
  }
  if (inherits(formula_raw, "formula")) {
    return(list(formula_raw))
  }
  if (is.list(formula_raw)) {
    return(lapply(formula_raw, function(f) {
      if (inherits(f, "formula")) f else as.formula(f)
    }))
  }
  stop("Unsupported formula specification.")
}

csv_path <- args[[1]]
output_json <- args[[2]]
formulas_json <- args[[3]]
family_name <- tolower(args[[4]])
method_name <- args[[5]]
select_flag <- tolower(args[[6]]) %in% c("true", "1", "yes")
test_name <- args[[7]]
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

formula_texts <- fromJSON(formulas_json)
formula_texts <- vapply(formula_texts, normalize_formula_text, character(1))

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
  gaulss = mgcv::gaulss(),
  gammals = mgcv::gammals(),
  ziplss = mgcv::ziplss(),
  gevlss = mgcv::gevlss(),
  shash = mgcv::shash(),
  stop(sprintf("Unsupported family for parity snapshot: %s", family_name))
)

fits <- lapply(
  formula_texts,
  function(txt) {
    gam(
      formula = coerce_formula_object(txt),
      data = data,
      family = family_obj,
      method = fit_method,
      select = select_flag
    )
  }
)

if (length(fits) == 1) {
  out <- anova(fits[[1]], freq = FALSE)
  payload <- list(
    parametric = if (is.null(out$pTerms.table)) NULL else list(
      labels = unname(as.character(rownames(out$pTerms.table))),
      values = unname(as.matrix(out$pTerms.table))
    ),
    smooth = if (is.null(out$s.table)) NULL else list(
      labels = unname(as.character(rownames(out$s.table))),
      values = unname(as.matrix(out$s.table))
    )
  )
} else {
  fit_args <- unname(fits)
  out <- do.call(anova, c(list(object = fit_args[[1]]), fit_args[-1], list(test = if (test_name == "NULL") NULL else test_name)))
  payload <- list(
    table = list(
      columns = colnames(out),
      values = unname(as.matrix(out))
    )
  )
}

write_json(payload, output_json, auto_unbox = TRUE, digits = 17, pretty = TRUE)
