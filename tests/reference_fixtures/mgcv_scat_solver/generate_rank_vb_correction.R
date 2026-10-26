# Opt-in source evidence: Rscript generate_rank_vb_correction.R INPUT OUTPUT
# Input is rank_input.json, frozen from the accepted SCAT workflow inputs.
suppressPackageStartupMessages(library(mgcv))
library(jsonlite)
a <- commandArgs(TRUE)
p <- fromJSON(a[1]); d <- as.data.frame(p$data)
trace('Vb.corr', where=asNamespace('mgcv'), print=FALSE, exit=quote({
  assign('.rank_capture', list(
    H=H, chol_ok=!inherits(R,'try-error'),
    R=if(inherits(R,'try-error')) NULL else R,
    correction=returnValue(), X=X, S=S, off=off, rho=rho, Vr=Vr,
    nth=nth, L=L, lsp0=lsp0, lambda=lambda, w=w), envir=.GlobalEnv)
}))
f <- do.call(scat,list(theta=p$theta,link=p$link,min.df=p$min_df))
b <- gam(as.formula(p$formula),data=d,weights=d$w,family=f,method='REML')
write_json(list(R_version=as.character(getRversion()),
  mgcv_version=as.character(packageVersion('mgcv')),
  capture=.rank_capture,theta=b$family$getTheta(),sp=b$sp,
  Vc=b$Vc,Vp=b$Vp,aic=AIC(b),modelX=predict(b,type='lpmatrix')),
  a[2],digits=17,auto_unbox=TRUE)
