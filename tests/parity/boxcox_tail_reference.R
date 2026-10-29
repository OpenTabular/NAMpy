#!/usr/bin/env Rscript
# BCT/BCPE probability-only references. Never invoked by ordinary tests.
a <- commandArgs(trailingOnly=TRUE)
if(length(a)!=1L || Sys.getenv('NAMPY_REFRESH_REFERENCE_FIXTURES')!='1') stop('Set NAMPY_REFRESH_REFERENCE_FIXTURES=1 and supply output directory')
lib <- Sys.getenv('GAMLSS_LIB_PATH',unset='')
if(nzchar(lib)) .libPaths(c(lib,.libPaths()))
suppressPackageStartupMessages(library(gamlss.dist))
out <- a[[1L]]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write.csv(data.frame(package=c('R','gamlss.dist'),version=c(as.character(getRversion()),as.character(packageVersion('gamlss.dist')))),file.path(out,'versions.csv'),row.names=FALSE)
# Preserve the installed boundary behavior: its nonpositive-response branches
# differ from the vendored pBCPE source. Tests replay the captured values only.
writeLines(c('# Installed gamlss.dist pBCPE probability implementation', 'pBCPE <-', deparse(pBCPE)),file.path(out,'pBCPE_source.R'))
mu <- c(.2,.7,1,2,5,3,1,1,1,1,2,2)
sigma <- c(.4,.8,1,1.2,.5,.7,2,2,1,.2,.4,.4)
nu <- c(-3,-2,-.5,0,.5,1,1,-1,0,2,.15,-.2)
y <- mu*c(.5,2,1.1,.3,1.5,.8,.2,5,2,.7,1.2,.8)
p <- c(.01,.1,.5,.9,.99,.8,.05,.95,.25,.75,.3,.6)
for(name in c('BCT','BCPE')) {
  tau <- if(name=='BCT') c(1,2,5,10,.7,30,1.5,4,8,20,5,5) else c(.7,1,2,3,4,.5,1.5,2,4,8,1,3)
  d <- get(paste0('d',name));pf <- get(paste0('p',name));q <- get(paste0('q',name))
  write.csv(data.frame(mu=mu,sigma=sigma,nu=nu,tau=tau,y=y,probability=p,
    log_prob=d(y,mu,sigma,nu,tau,log=TRUE),cdf=pf(y,mu,sigma,nu,tau),
    survival=pf(y,mu,sigma,nu,tau,lower.tail=FALSE),
    log_cdf=pf(y,mu,sigma,nu,tau,log.p=TRUE),
    log_survival=pf(y,mu,sigma,nu,tau,lower.tail=FALSE,log.p=TRUE),
    quantile=q(p,mu,sigma,nu,tau)),file.path(out,paste0(tolower(name),'.csv')),row.names=FALSE)
  yy <- c(-1,0,1e-200,1e200)
  nn <- c(1,1,1,-1)
  tt <- if(name=='BCT') 4 else 1.5
  suppressWarnings(write.csv(data.frame(mu=1,sigma=1,nu=nn,tau=tt,y=yy,
    log_prob=d(yy,1,1,nn,tt,log=TRUE),log_cdf=pf(yy,1,1,nn,tt,log.p=TRUE),
    log_survival=pf(yy,1,1,nn,tt,lower.tail=FALSE,log.p=TRUE)),file.path(out,paste0(tolower(name),'_quirks.csv')),row.names=FALSE))
}
write.csv(data.frame(mu=1,sigma=1,nu=0,tau=2e6,y=exp(100),
    log_prob=dBCT(exp(100),1,1,0,2e6,log=TRUE),
    gaussian_log_prob=dBCCG(exp(100),1,1,0,log=TRUE)),file.path(out,'bct_density_switch.csv'),row.names=FALSE)
write.csv(data.frame(mu=1,sigma=1,nu=0,tau=1000,y=exp(.02),probability=.51,
    cdf=pBCPE(exp(.02),1,1,0,1000),quantile=qBCPE(.51,1,1,0,1000)),
    file.path(out,'bcpe_gamma_underflow.csv'),row.names=FALSE)
