#!/usr/bin/env Rscript
# Fixed-probability and unpenalized BB reference; opt-in refresh only.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 1L || Sys.getenv("NAMPY_REFRESH_REFERENCE_FIXTURES") != "1")
  stop("usage: NAMPY_REFRESH_REFERENCE_FIXTURES=1 Rscript betabinomial_lss_reference.R OUTPUT_DIR")
lib <- Sys.getenv("GAMLSS_LIB_PATH", unset="")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(gamlss.dist))
out <- args[[1L]]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
write.csv(data.frame(package=c("R", "gamlss", "gamlss.dist"),
                     version=c(as.character(getRversion()), as.character(packageVersion("gamlss")), as.character(packageVersion("gamlss.dist")))),
          file.path(out, "versions.csv"), row.names=FALSE)
mu <- c(.1, .3, .5, .7, .9, .4)
kappa <- c(.5, 2, 5, 1.5, 8, 3)
trials <- c(1, 4, 10, 15, 12, 8)
y <- c(0, 1, 5, 15, 10, 0)
probability <- c(.01, .1, .5, .9, .99, .2)
write.csv(data.frame(mu=mu, kappa=kappa, trials=trials, y=y, probability=probability,
                     log_prob=dBB(y, mu=mu, sigma=1/kappa, bd=trials, log=TRUE),
                     cdf=pBB(y, mu=mu, sigma=1/kappa, bd=trials),
                     quantile=qBB(probability, mu=mu, sigma=1/kappa, bd=trials)),
          file.path(out, "probability.csv"), row.names=FALSE)
# sigma is deliberately far from the upstream binomial-approximation cutoff.
set.seed(42)
n <- 200
x <- seq(-1, 1, length.out=n)
trials <- rep(c(3, 5, 8, 12, 16), length.out=n)
y <- rBB(n, mu=plogis(-.2+.5*x), sigma=.4, bd=trials)
data <- data.frame(x=x, y=y, trials=trials)
fit <- gamlss(cbind(y, trials-y)~x, sigma.formula=~1, family=BB(), data=data,
              control=gamlss.control(n.cyc=1000L, c.crit=1e-10, trace=FALSE))
if (!fit$converged) stop("BB reference fit did not converge")
data$mu <- fitted(fit, "mu")
data$kappa <- 1/fitted(fit, "sigma")
data$log_prob <- dBB(y, mu=data$mu, sigma=1/data$kappa, bd=trials, log=TRUE)
write.csv(data, file.path(out, "fit.csv"), row.names=FALSE)
