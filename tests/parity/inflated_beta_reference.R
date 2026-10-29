#!/usr/bin/env Rscript
# Fixed BEINF probabilities and unpenalized semantic reference, opt-in only.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=1L || Sys.getenv("NAMPY_REFRESH_REFERENCE_FIXTURES")!="1") stop("Set NAMPY_REFRESH_REFERENCE_FIXTURES=1 and supply output directory")
lib <- Sys.getenv("GAMLSS_LIB_PATH",unset="")
if(nzchar(lib)) .libPaths(c(lib,.libPaths()))
suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(gamlss.dist))
out <- args[[1L]]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write.csv(data.frame(package=c("R","gamlss","gamlss.dist"),version=c(as.character(getRversion()),as.character(packageVersion("gamlss")),as.character(packageVersion("gamlss.dist")))),file.path(out,"versions.csv"),row.names=FALSE)
mu <- c(.1,.3,.5,.7,.9,.4)
phi <- c(2,4,6,3,8,1.5)
nu <- c(.2,.4,.7,1.5,2,.3)
tau <- c(.3,.8,1.2,.2,1,.6)
sigma <- 1/sqrt(1+phi)
pc <- 1/(1+nu+tau)
p0 <- nu*pc
p1 <- tau*pc
y <- c(0,.1,.5,1,.8,.4)
probability <- c(.01,.1,.5,.9,.99,.8)
mean <- p1+pc*mu
mixture_variance <- pc*(mu*(1-mu)/(1+phi)+(mu-mean)^2)+p0*mean^2+p1*(1-mean)^2
write.csv(data.frame(mu=mu,phi=phi,pi0=p0,pi1=p1,pi_interior=pc,y=y,probability=probability,
 log_prob=dBEINF(y,mu,sigma,nu,tau,log=TRUE),cdf=pBEINF(y,mu,sigma,nu,tau),quantile=qBEINF(probability,mu,sigma,nu,tau),
 mixture_variance=mixture_variance,reported_variance=BEINF()$variance(mu,sigma,nu,tau)),file.path(out,"probability.csv"),row.names=FALSE)
set.seed(612)
x <- seq(-1,1,length.out=240)
y <- rBEINF(length(x),mu=plogis(-.3+.4*x),sigma=1/sqrt(7),nu=exp(-1+.3*x),tau=exp(-1.2-.2*x))
data <- data.frame(x=x,y=y)
fit <- gamlss(y~x,sigma.formula=~1,nu.formula=~x,tau.formula=~x,family=BEINF(),data=data,
 control=gamlss.control(n.cyc=1000L,c.crit=1e-10,trace=FALSE))
if(!fit$converged) stop("BEINF reference failed to converge")
data$mu <- fitted(fit,"mu")
data$phi <- 1/fitted(fit,"sigma")^2-1
nu <- fitted(fit,"nu")
tau <- fitted(fit,"tau")
data$pi_interior <- 1/(1+nu+tau)
data$pi0 <- nu*data$pi_interior
data$pi1 <- tau*data$pi_interior
write.csv(data,file.path(out,"fit.csv"),row.names=FALSE)
