#!/usr/bin/env Rscript
# BCCG probability and source-hook references; ordinary tests never invoke R.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=1L || Sys.getenv("NAMPY_REFRESH_REFERENCE_FIXTURES")!="1") stop("Set NAMPY_REFRESH_REFERENCE_FIXTURES=1 and supply output directory")
lib <- Sys.getenv("GAMLSS_LIB_PATH",unset="")
if(nzchar(lib)) .libPaths(c(lib,.libPaths()))
suppressPackageStartupMessages(library(gamlss.dist))
out <- args[[1L]]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write.csv(data.frame(package=c("R","gamlss.dist"),version=c(as.character(getRversion()),as.character(packageVersion("gamlss.dist")))),file.path(out,"versions.csv"),row.names=FALSE)
mu <- c(.2,.7,1,2,5,3,1,1,1,1)
sigma <- c(.4,.8,1,1.2,.5,.7,2,2,1,.2)
nu <- c(-3,-2,-.5,0,.5,1,1,-1,0,2)
y <- mu*c(.5,2,1.1,.3,1.5,.8,.2,5,2,.7)
probability <- c(.01,.1,.5,.9,.99,.8,.05,.95,.25,.75)
write.csv(data.frame(mu=mu,sigma=sigma,nu=nu,y=y,probability=probability,
 log_prob=dBCCG(y,mu,sigma,nu,log=TRUE),cdf=pBCCG(y,mu,sigma,nu),survival=pBCCG(y,mu,sigma,nu,lower.tail=FALSE),
 quantile=qBCCG(probability,mu,sigma,nu)),file.path(out,"probability.csv"),row.names=FALSE)
# Record upstream support/near-zero quirks independently from ordinary semantics.
q <- c(-1,0,1e-200,1e200)
mu <- rep(1,4)
sigma <- rep(1,4)
nu <- c(1,1,1,-1)
suppressWarnings(write.csv(data.frame(mu=mu,sigma=sigma,nu=nu,y=q,
 log_prob=dBCCG(q,mu,sigma,nu,log=TRUE),log_cdf=pBCCG(q,mu,sigma,nu,log.p=TRUE),
 log_survival=pBCCG(q,mu,sigma,nu,lower.tail=FALSE,log.p=TRUE)),file.path(out,"source_quirks.csv"),row.names=FALSE))
# Scores are natural-parameter derivatives. Second hooks are GAMLSS working
# curvature, deliberately retained separately from an observed Hessian oracle.
mu <- c(.7,1.2,2,3,1)
sigma <- c(.4,.8,1.2,.6,.4)
nu <- c(-2,-.5,1,2,0)
y <- c(.4,2,.7,4,1.3)
f <- BCCG()
write.csv(data.frame(mu=mu,sigma=sigma,nu=nu,y=y,
 score_mu=f$dldm(y,mu,sigma,nu),score_sigma=f$dldd(y,mu,sigma,nu),score_nu=f$dldv(y,mu,sigma,nu),
 working_mu_mu=f$d2ldm2(y,mu,sigma,nu),working_sigma_sigma=f$d2ldd2(sigma),
 working_nu_nu=f$d2ldv2(sigma),working_mu_sigma=f$d2ldmdd(mu,sigma,nu),
 working_mu_nu=f$d2ldmdv(mu),working_sigma_nu=f$d2ldddv(sigma,nu)),
 file.path(out,"derivative_hooks.csv"),row.names=FALSE)
