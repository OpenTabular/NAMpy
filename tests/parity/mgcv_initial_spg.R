# Usage:
#   Rscript mgcv_initial_spg.R <csv_path> <output_json> <formula> <family> <method> <select>
#
# Builds mgcv's `gam(..., fit = FALSE)` setup and returns `initial.spg(...)`
# for parity with NAMpy's `_initial_smoothing_params_mgcv_style()`.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop(
    "Usage: Rscript mgcv_initial_spg.R <csv_path> <output_json> <formula> <family> <method> <select>"
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

csv_path <- args[[1]]
output_json <- args[[2]]
formula_text <- normalize_formula_text(args[[3]])
family_name <- tolower(args[[4]])
method_name <- args[[5]]
select_flag <- tolower(args[[6]]) %in% c("true", "1", "yes")

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
  binomial = {
    link <- if (is.null(family_param) || family_param == "") "logit" else family_param
    binomial(link = link)
  },
  poisson = poisson(link = "log"),
  gamma = {
    link <- if (is.null(family_param) || family_param == "") "inverse" else family_param
    Gamma(link = link)
  },
  negbin = {
    theta <- if (is.null(family_param)) 1.0 else as.numeric(family_param)
    mgcv::nb(theta = theta, link = "log")
  },
  negbin_est = {
    theta <- if (is.null(family_param)) 1.0 else as.numeric(family_param)
    mgcv::nb(theta = -abs(theta), link = "log")
  },
  gaulss = mgcv::gaulss(),
  gammals = mgcv::gammals(),
  stop(sprintf("Unsupported family for initial.spg parity: %s", family_name))
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

G <- prefit

if (inherits(G$family, "extended.family") && inherits(G$family, "general.family")) {
  G$Sl <- mgcv:::Sl.setup(G)
  G$X <- mgcv:::Sl.initial.repara(G$Sl, G$X, both.sides = FALSE)
}

G$family <- mgcv:::fix.family(G$family)
G$rS <- mgcv:::mini.roots(G$S, G$off, ncol(G$X), G$rank)
Ssp <- mgcv:::totalPenaltySpace(G$S, G$H, G$off, ncol(G$X))
G$Eb <- Ssp$E
G$U1 <- cbind(Ssp$Y, Ssp$Z)
G$Mp <- ncol(Ssp$Z)

if (!is.null(G$family$preinitialize)) {
  if (inherits(G$family, "general.family")) {
    Gmod <- G$family$preinitialize(G)
    for (gnam in names(Gmod)) G[[gnam]] <- Gmod[[gnam]]
  } else {
    if (!is.null(attr(G$family$preinitialize, "needG"))) attr(G$family, "G") <- G
    pini <- G$family$preinitialize(G$y, G$family)
    attr(G$family, "G") <- NULL
    if (!is.null(pini$family)) G$family <- pini$family
    if (!is.null(pini$Theta)) G$family$putTheta(pini$Theta)
    if (!is.null(pini$y)) G$y <- pini$y
  }
}

initial_sp <- mgcv:::initial.spg(
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
)

start <- NULL
x <- G$X
y <- G$y
weights <- G$w
family <- G$family
offset <- G$offset
E <- G$Eb
nobs <- nrow(x)
pen.reg <- get("pen.reg", envir = asNamespace("mgcv"))
eval(family$initialize)
lbb <- family$ll(y, x, start, weights, family, offset = offset, deriv = 1)$lbb

payload <- list(
  initial_sp = unname(as.numeric(initial_sp)),
  start = unname(as.numeric(start)),
  lbb = unname(lbb),
  X_initial = unname(G$X),
  S = lapply(G$S, function(Si) unname(Si)),
  rank = unname(as.integer(G$rank)),
  off = unname(as.integer(G$off)),
  Eb = unname(G$Eb),
  Sl_blocks = lapply(G$Sl, function(block) {
    list(
      start = as.integer(block$start),
      stop = as.integer(block$stop),
      rank = if (is.null(block$rank)) NULL else as.integer(block$rank),
      repara = isTRUE(block$repara),
      linear = if (is.null(block$linear)) TRUE else isTRUE(block$linear),
      nS = length(block$S),
      ind = if (is.null(block$ind)) NULL else unname(as.logical(block$ind)),
      D = if (is.null(block$D)) NULL else unname(block$D),
      Di = if (is.null(block$Di)) NULL else unname(block$Di),
      S = lapply(block$S, function(Si) unname(Si))
    )
  })
)

write_json(payload, output_json, auto_unbox = TRUE, digits = 17, pretty = TRUE)
