# Probe mgcv::initial.spg for general-family cases.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop("Usage: Rscript debug/mgcv_general_initial_spg.R <csv> <formula> <family> <method> <select>")
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
    lapply(x, function(f) if (inherits(f, "formula")) f else as.formula(f))
  } else {
    stop("Unsupported formula specification.")
  }
}

suppressPackageStartupMessages(library(mgcv))

data <- read.csv(args[[1]], stringsAsFactors = FALSE)
formula_text <- normalize_formula_text(args[[2]])
formula_raw <- NULL
if (grepl("^\\s*(c|list)\\s*\\(", formula_text)) {
  formula_raw <- eval(parse(text = formula_text))
}
formula_obj <- if (is.null(formula_raw)) as.formula(formula_text) else coerce_formula_list(formula_raw)
family_obj <- switch(tolower(args[[3]]), gaulss = mgcv::gaulss(), gammals = mgcv::gammals())
method_name <- args[[4]]
select_flag <- tolower(args[[5]]) %in% c("true", "1", "yes")

G <- gam(
  formula = formula_obj,
  data = data,
  family = family_obj,
  method = method_name,
  select = select_flag,
  fit = FALSE
)
G$family <- mgcv:::fix.family(G$family)
G$Sl <- mgcv:::Sl.setup(G)
G$X <- mgcv:::Sl.initial.repara(G$Sl, G$X, both.sides = FALSE)
G$rS <- mgcv:::mini.roots(G$S, G$off, ncol(G$X), G$rank)
Ssp <- mgcv:::totalPenaltySpace(G$S, G$H, G$off, ncol(G$X))
G$Eb <- Ssp$E
G$U1 <- cbind(Ssp$Y, Ssp$Z)
G$Mp <- ncol(Ssp$Z)
if (!is.null(G$family$preinitialize)) {
  Gmod <- G$family$preinitialize(G)
  for (gnam in names(Gmod)) G[[gnam]] <- Gmod[[gnam]]
}
lsp <- log(mgcv:::initial.spg(
  G$X,
  G$y,
  G$w,
  G$family,
  G$S,
  G$rank,
  G$off,
  offset = G$offset,
  L = G$L,
  lsp0 = G$lsp0,
  E = G$Eb
))
cat(paste(format(exp(lsp), digits = 17), collapse = " "), "\n")
cat(paste(format(lsp, digits = 17), collapse = " "), "\n")
