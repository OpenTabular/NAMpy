#!/usr/bin/env Rscript
# Explicit refresh only. Ordinary tests consume saved BCCG endpoints.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=1L || Sys.getenv('NAMPY_REFRESH_REFERENCE_FIXTURES')!='1') stop('Set NAMPY_REFRESH_REFERENCE_FIXTURES=1 and supply output directory')
lib <- Sys.getenv('GAMLSS_LIB_PATH',unset='')
if(nzchar(lib)) .libPaths(c(lib,.libPaths()))
suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(gamlss.dist))
out <- args[[1L]]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write.csv(data.frame(package=c('R','gamlss','gamlss.dist'),version=c(as.character(getRversion()),as.character(packageVersion('gamlss')),as.character(packageVersion('gamlss.dist')))),file.path(out,'versions.csv'),row.names=FALSE)
x <- seq(-1,1,length.out=180)
p <- (rank(sin(seq_along(x)*1.7)+cos(seq_along(x)*.41))-.5)/length(x)
for(case in c('positive','negative','weighted')) {
    nu <- if(case=='negative') -3 else .7
    sigma <- if(case=='negative') .18 else .28
    y <- qBCCG(p,mu=exp(.3+.2*x),sigma=sigma,nu=nu)
    w <- if(case=='weighted') rep(c(0,1,2,.5,3),length.out=length(x)) else rep(1,length(x))
    data <- data.frame(x=x,y=y,w=w)
    fit <- gamlss(y~x,sigma.formula=~1,nu.formula=~1,
        family=BCCG(mu.link='log'),data=data,weights=w,
        control=gamlss.control(n.cyc=2000L,c.crit=1e-11,trace=FALSE))
    if(!fit$converged) stop(paste('Reference did not converge:',case))
    data$mu <- fitted(fit,'mu')
    data$sigma <- fitted(fit,'sigma')
    data$nu <- fitted(fit,'nu')
    data$log_prob <- dBCCG(y,data$mu,data$sigma,data$nu,log=TRUE)
    write.csv(data,file.path(out,paste0(case,'.csv')),row.names=FALSE)
}
