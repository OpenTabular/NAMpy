suppressPackageStartupMessages(library(mgcv))

args <- commandArgs(trailingOnly = TRUE)
csv_path <- args[[1]]
family_name <- args[[2]]
formula_json <- args[[3]]

data <- read.csv(csv_path)
formula_parts <- jsonlite::fromJSON(formula_json)
form <- lapply(formula_parts, as.formula)
fam <- switch(
  family_name,
  gevlss = gevlss(),
  stop("unsupported family")
)
fit <- gam(form, data = data, family = fam, method = "ML")
X_model <- model.matrix(fit)
X_lp <- predict(fit, type = "lpmatrix")
cat("dim model", paste(dim(X_model), collapse = "x"), "\n")
cat("dim lpmatrix", paste(dim(X_lp), collapse = "x"), "\n")
cat("max abs model-lpmatrix", max(abs(X_model - X_lp)), "\n")
cat("coef names", paste(names(coef(fit)), collapse = " | "), "\n")
for (sm in fit$smooth) {
  cat("smooth", sm$label, sm$first.para, sm$last.para, "\n")
}
print(concurvity(fit, full = TRUE))
X <- qr.R(qr(X_model, tol = 0, LAPACK = FALSE))
start <- c(1, vapply(fit$smooth, function(sm) sm$first.para, numeric(1)))
stop <- c(min(start[-1]) - 1, vapply(fit$smooth, function(sm) sm$last.para, numeric(1)))
for (i in seq_along(start)) {
  keep <- rep(TRUE, ncol(X))
  keep[start[i]:stop[i]] <- FALSE
  Xi <- X[, keep, drop = FALSE]
  Xj <- X[, start[i]:stop[i], drop = FALSE]
  q <- qr(cbind(Xi, Xj), LAPACK = FALSE, tol = 0)
  cat("inner pivot", i, paste(q$pivot, collapse = ","), "\n")
}
