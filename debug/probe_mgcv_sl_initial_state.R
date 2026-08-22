suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop("Usage: Rscript probe_mgcv_sl_initial_state.R <csv> <formula> <family> <method> <output>")
}

csv_path <- args[[1]]
formula_text <- args[[2]]
family_name <- tolower(args[[3]])
method_name <- args[[4]]
output_json <- args[[5]]

coerce_formula <- function(x) {
  obj <- eval(parse(text = x))
  if (is.character(obj)) {
    if (length(obj) == 1) return(as.formula(obj))
    return(lapply(obj, as.formula))
  }
  obj
}

data <- read.csv(csv_path, stringsAsFactors = FALSE)
for (nm in names(data)) if (is.character(data[[nm]])) data[[nm]] <- factor(data[[nm]])

family_obj <- switch(
  family_name,
  gaulss = mgcv::gaulss(),
  gammals = mgcv::gammals(),
  gevlss = mgcv::gevlss(),
  stop(sprintf("Unsupported family: %s", family_name))
)

G <- gam(
  formula = coerce_formula(formula_text),
  data = data,
  family = family_obj,
  method = method_name,
  fit = FALSE
)

X_before <- G$X
S_before <- G$S
Sl <- mgcv:::Sl.setup(G)
X_after <- mgcv:::Sl.initial.repara(Sl, G$X, both.sides = FALSE)

blocks <- lapply(Sl, function(block) {
  list(
    start = unname(block$start),
    stop = unname(block$stop),
    rank = unname(block$rank),
    repara = unname(block$repara),
    linear = unname(block$linear),
    ind = if (is.null(block$ind)) NULL else unname(block$ind),
    D = if (is.null(block$D)) NULL else unname(block$D),
    Di = if (is.null(block$Di)) NULL else unname(block$Di),
    S = lapply(block$S, unname)
  )
})

write_json(
  list(
    X_before = unname(X_before),
    X_after = unname(X_after),
    S_before = lapply(S_before, unname),
    off = unname(G$off),
    rank = unname(G$rank),
    blocks = blocks
  ),
  output_json,
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE
)
