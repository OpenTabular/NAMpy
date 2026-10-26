# Fixed late NB optim state: gam.fit4 obtains P through C_gdi2/get_bSb.
library(mgcv);library(jsonlite)
a<-commandArgs(TRUE);d<-as.data.frame(fromJSON(a[1]));captured<-NULL
trace('gam.fit4',where=asNamespace('mgcv'),print=FALSE,exit=quote({
 rv<-returnValue()
 captured<<-list(criterion=rv$REML,penalty=oo$P,deviance=dev,logdet=oo$ldet,
                scale=scale,log_theta=theta,log_sp=sp,coef=rv$coefficients)
}))
b<-gam(y~s(x0,bs='cr',k=8),data=d,family=nb(theta=exp(-0.12400663050520905)),
       sp=exp(20.20394495173847),method='REML')
untrace('gam.fit4',where=asNamespace('mgcv'))
write_json(list(R=as.character(getRversion()),mgcv=as.character(packageVersion('mgcv')),
 captured=captured,score=as.numeric(b$gcv.ubre),coef=coef(b)),a[2],digits=17,auto_unbox=TRUE)
