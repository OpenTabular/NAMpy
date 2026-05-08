args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript mgcv_t2_vc_breakdown.R <csv_path> <output_json> <formula>")
}

normalize_formula_text <- function(x) {
  x <- gsub("\\[", "c(", x)
  x <- gsub("\\]", ")", x)
  x <- gsub("\\bTrue\\b", "TRUE", x)
  x <- gsub("\\bFalse\\b", "FALSE", x)
  x <- gsub("\\bNone\\b", "NULL", x)
  x
}

suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))

csv_path <- args[[1]]
output_json <- args[[2]]
formula_obj <- as.formula(normalize_formula_text(args[[3]]))
data <- read.csv(csv_path, stringsAsFactors = FALSE)

family_obj <- gaussian()
family_obj <- mgcv:::fix.family.link(family_obj)
family_obj <- mgcv:::fix.family.var(family_obj)
family_obj <- mgcv:::fix.family.ls(family_obj)

fit <- gam(formula_obj, data = data, family = family_obj, method = "REML")
G <- gam(formula_obj, data = data, family = family_obj, method = "REML", fit = FALSE)
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

fit3 <- mgcv:::gam.fit3(
  x = G$X,
  y = G$y,
  sp = log(pmax(unname(fit$sp), 1e-300)),
  Eb = G$Eb,
  UrS = G$UrS,
  weights = G$w,
  offset = G$offset,
  U1 = G$U1,
  Mp = G$Mp,
  family = G$family,
  control = gam.control(),
  intercept = TRUE,
  deriv = 2,
  gamma = 1,
  scale = 1,
  scoreType = "REML",
  null.coef = rep(0, ncol(G$X)),
  pearson.extra = G$pearson.extra,
  dev.extra = G$dev.extra,
  n.true = G$n.true
)
fit3$outer.info <- fit$outer.info
fit3$sp <- fit$sp
fit3$method <- fit$method
fit3$scale <- fit$scale
fit3$scale.est <- fit$scale
fit3$scale.estimated <- TRUE
fit3$control <- gam.control()
fit3$control$nthreads <- 1

breakdown <- function(X, L, lsp0, S, off, object, gamma) {
  scale <- if (object$scale.estimated) object$scale.est else object$scale
  Vb <- object$rV %*% t(object$rV) * scale
  WX <- sqrt(object$weights) * X
  qrx <- mgcv:::pqr(WX, object$control$nthreads)
  R <- mgcv:::pqr.R(qrx)
  R[, qrx$pivot] <- R

  hess <- object$outer.info$hess
  db.drho <- object$db.drho
  dw.drho <- object$dw.drho
  lsp <- log(object$sp)
  M <- ncol(db.drho)
  if (!is.null(L)) {
    db.drho <- db.drho %*% L[1:M, , drop = FALSE]
    M <- ncol(db.drho)
  }
  ev <- eigen(hess, symmetric = TRUE)
  d <- ev$values
  ind <- d <= 0
  d[ind] <- 0
  d[!ind] <- 1 / sqrt(d[!ind])
  rV <- (d * t(ev$vectors))[, 1:M]
  Vc1 <- crossprod(rV %*% t(db.drho))
  d <- ev$values
  d[ind] <- 0
  d <- 1 / sqrt(d + 1 / 10)
  Vr <- crossprod(d * t(ev$vectors))
  nth <- if (is.null(object$family$n.theta)) 0 else object$family$n.theta
  drop.scale <- object$scale.estimated && !(object$method %in% c("P-REML", "P-ML"))
  Vc2 <- scale * mgcv:::Vb.corr(R, L, lsp0, S, off, dw.drho, w = NULL, lsp, Vr, nth, drop.scale)
  Vc <- Vb + Vc1 + Vc2
  list(
    scale = unname(as.numeric(scale)),
    X = unname(as.matrix(X)),
    R = unname(as.matrix(R)),
    Vb = unname(as.matrix(Vb)),
    Vc1 = unname(as.matrix(Vc1)),
    Vc2 = unname(as.matrix(Vc2)),
    Vc = unname(as.matrix(Vc)),
    db_drho = unname(as.matrix(db.drho)),
    hess = unname(as.matrix(hess)),
    dw_drho = unname(as.matrix(dw.drho)),
    final_Vc = unname(as.matrix(fit$Vc)),
    final_Vp = unname(as.matrix(fit$Vp)),
    final_coef = unname(as.numeric(fit$coefficients)),
    final_lpmatrix_train = unname(as.matrix(predict(fit, type = "lpmatrix"))),
    G_P = if (is.null(G$P)) NULL else unname(as.matrix(G$P)),
    fit3_coef = unname(as.numeric(fit3$coefficients)),
    sp = unname(as.numeric(fit$sp))
  )
}

payload <- breakdown(G$X, G$L, G$lsp0, G$S, G$off, fit3, 1)
write_json(payload, output_json, auto_unbox = TRUE, digits = 17, pretty = TRUE, null = "null")
