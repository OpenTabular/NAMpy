#!/usr/bin/env Rscript
# Small fixed-parameter and unpenalized LOGNO/WEI references. Normal tests
# consume the saved CSVs; no R process is invoked by their reader.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 1L || Sys.getenv("NAMPY_REFRESH_REFERENCE_FIXTURES") != "1") {
  stop("usage: NAMPY_REFRESH_REFERENCE_FIXTURES=1 Rscript positive_lss_reference.R OUTPUT_DIR")
}
lib <- Sys.getenv("GAMLSS_LIB_PATH", unset="")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(gamlss.dist))
out <- args[[1L]]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
versions <- data.frame(package=c("R", "gamlss", "gamlss.dist"),
                       version=c(as.character(getRversion()), as.character(packageVersion("gamlss")), as.character(packageVersion("gamlss.dist"))))
write.csv(versions, file.path(out, "versions.csv"), row.names=FALSE)
probability <- c(0.01, 0.1, 0.5, 0.9, 0.99)
for (name in c("lognormal", "weibull")) {
  first <- if (name == "lognormal") c(-1.5, -0.4, 0.2, 1, 2) else c(0.2, 0.7, 1.2, 2, 5)
  second <- c(0.35, 0.7, 1.1, 1.8, 3.1)
  y <- c(0.15, 0.5, 1.2, 3, 8)
  family <- if (name == "lognormal") "LOGNO" else "WEI"
  density <- get(paste0("d", family), asNamespace("gamlss.dist"))
  cdf <- get(paste0("p", family), asNamespace("gamlss.dist"))
  quantile <- get(paste0("q", family), asNamespace("gamlss.dist"))
  values <- data.frame(first=first, second=second, y=y, probability=probability,
                       log_density=density(y, first, second, log=TRUE),
                       cdf=cdf(y, first, second),
                       quantile=quantile(probability, first, second))
  write.csv(values, file.path(out, paste0(name, "_probability.csv")), row.names=FALSE)
  x <- seq(-1, 1, length.out=120)
  phase <- seq_along(x)
  u <- (rank(sin(phase*1.7)+cos(phase*0.41))-0.5)/length(x)
  response <- if (name == "lognormal") exp(0.5+0.2*x+0.45*qnorm(u)) else exp(0.2+0.3*x)*(-log(1-u))^(1/1.8)
  data <- data.frame(x=x, y=response)
  fitted <- gamlss(y~x, sigma.formula=~1, family=get(family, asNamespace("gamlss.dist"))(), data=data,
                   control=gamlss.control(n.cyc=1000L, c.crit=1e-10, trace=FALSE))
  if (!fitted$converged) stop(paste("Reference did not converge:", family))
  data$first <- fitted(fitted, "mu")
  data$second <- fitted(fitted, "sigma")
  data$log_density <- density(data$y, data$first, data$second, log=TRUE)
  write.csv(data, file.path(out, paste0(name, "fit.csv")), row.names=FALSE)
}
