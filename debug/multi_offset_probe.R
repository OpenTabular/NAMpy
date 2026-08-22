# Probe: what does mgcv::gam do with two offset() terms in one formula?
# interpret.gam0 (mgcv/R/mgcv.r:387-389) assigns all offset labels into the
# single slot av[kp]; R keeps the first element with a warning, so only the
# first offset should reach model.offset(). This probe pins that behavior.
library(mgcv)
set.seed(1)
n <- 80
d <- data.frame(x = runif(n), a = runif(n), b = runif(n))
d$y <- rpois(n, exp(1 + sin(2 * d$x) + d$a + d$b))

fit2 <- tryCatch(
  withCallingHandlers(
    gam(y ~ offset(a) + offset(b) + s(x, bs = "cr", k = 8),
        family = poisson(), data = d, method = "REML"),
    warning = function(w) {
      cat("WARNING:", conditionMessage(w), "\n")
      invokeRestart("muffleWarning")
    }
  ),
  error = function(e) {
    cat("ERROR:", conditionMessage(e), "\n")
    NULL
  }
)
if (!is.null(fit2)) {
  fit_a <- gam(y ~ offset(a) + s(x, bs = "cr", k = 8),
               family = poisson(), data = d, method = "REML")
  d$ab <- d$a + d$b
  fit_ab <- gam(y ~ offset(ab) + s(x, bs = "cr", k = 8),
                family = poisson(), data = d, method = "REML")
  cat("max|fit2 - fit_first_only|  =",
      max(abs(fitted(fit2) - fitted(fit_a))), "\n")
  cat("max|fit2 - fit_sum|         =",
      max(abs(fitted(fit2) - fitted(fit_ab))), "\n")
  cat("max|offset - a|             =", max(abs(fit2$offset - d$a)), "\n")
  cat("max|offset - (a+b)|         =", max(abs(fit2$offset - d$ab)), "\n")
}
