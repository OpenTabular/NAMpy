# Usage:
#   Rscript mgcv_mrf_inner_state.R <csv_path> <output_json> <formula> <sp_json>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop(
    "Usage: Rscript mgcv_mrf_inner_state.R <csv_path> <output_json> <formula> <sp_json>"
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

csv_path <- args[[1]]
output_json <- args[[2]]
formula_text <- normalize_formula_text(args[[3]])
sp <- as.numeric(jsonlite::fromJSON(args[[4]]))

suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))

data <- read.csv(csv_path, stringsAsFactors = FALSE)
for (nm in names(data)) {
  if (is.character(data[[nm]])) data[[nm]] <- factor(data[[nm]])
}

family_obj <- mgcv:::fix.family(gaussian())
family_obj <- mgcv:::fix.family.link(family_obj)
family_obj <- mgcv:::fix.family.var(family_obj)
family_obj <- mgcv:::fix.family.ls(family_obj)

G <- gam(
  formula = as.formula(formula_text),
  data = data,
  family = family_obj,
  method = "REML",
  sp = unname(sp),
  fit = FALSE
)

G$family <- mgcv:::fix.family(G$family)
G$rS <- mgcv:::mini.roots(G$S, G$off, ncol(G$X), G$rank)
Ssp <- mgcv:::totalPenaltySpace(G$S, G$H, G$off, ncol(G$X))
G$Eb <- Ssp$E
G$U1 <- cbind(Ssp$Y, Ssp$Z)
G$Mp <- ncol(Ssp$Z)
G$UrS <- list()
if (length(G$S) > 0) {
  for (i in seq_along(G$S)) {
    G$UrS[[i]] <- t(Ssp$Y) %*% G$rS[[i]]
  }
}

x <- as.matrix(G$X)
q <- ncol(x)
rp <- mgcv:::gam.reparam(G$UrS, log(pmax(unname(sp), 1e-300)), 0)

T <- diag(q)
if (ncol(rp$Qs) > 0) {
  T[1:ncol(rp$Qs), 1:ncol(rp$Qs)] <- rp$Qs
}
T <- G$U1 %*% T
x_curr <- x %*% T

rS_curr <- list()
if (length(G$UrS) > 0) {
  for (i in seq_along(G$UrS)) {
    rS_curr[[i]] <- rbind(rp$rS[[i]], matrix(0, G$Mp, ncol(rp$rS[[i]])))
  }
}
Eb_curr <- G$Eb %*% T
rows_E <- q - G$Mp
Sr_curr <- cbind(rp$E, matrix(0, nrow(rp$E), G$Mp))

weights <- as.numeric(G$w)
y <- as.numeric(G$y)
n <- length(y)
zg <- rep(0.0, max(n, q))
zg[1:n] <- y

oo <- .C(
  "pls_fit1",
  PACKAGE = "mgcv",
  y = as.double(zg),
  X = as.double(x_curr),
  w = as.double(weights),
  wy = as.double(weights * y),
  E = as.double(Sr_curr),
  Es = as.double(Eb_curr),
  n = as.integer(n),
  q = as.integer(q),
  rE = as.integer(rows_E),
  eta = as.double(y),
  penalty = as.double(1),
  rank.tol = as.double(.Machine$double.eps * 100),
  nt = as.integer(1),
  use.wy = as.integer(0)
)

coef_curr <- unname(as.numeric(oo$y[seq_len(q)]))
eta_inner <- unname(as.numeric(drop(x_curr %*% coef_curr)))
dev_inner <- unname(as.numeric(sum(G$family$dev.resids(y, eta_inner, weights))))

payload <- list(
  coef_current = coef_curr,
  eta_inner = eta_inner,
  deviance_inner = dev_inner,
  x_current = unname(x_curr),
  T = unname(T),
  Sr_current = unname(Sr_curr),
  Eb_current = unname(Eb_curr),
  use_wy = as.integer(oo$use.wy),
  penalty = unname(as.numeric(oo$penalty))
)

write_json(
  payload,
  output_json,
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
