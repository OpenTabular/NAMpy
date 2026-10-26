library(mgcv);library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1]);d<-as.data.frame(p$data)
trace('gam.fit4',where=asNamespace('mgcv'),print=FALSE,tracer=quote({
 if(!exists('.captured',envir=.GlobalEnv,inherits=FALSE)) {
  nn<-setdiff(names(formals(mgcv:::gam.fit4)), '...')
  aa<-lapply(nn,function(n)get(n,envir=environment(),inherits=TRUE));names(aa)<-nn
  assign('.captured',aa,envir=.GlobalEnv)
  stop('captured first call')
 }
}))
try(gam(as.formula(p$formula),data=d,family=scat(theta=p$theta),weights=w,sp=p$sp,scale=2,method='REML'),silent=TRUE)
untrace('gam.fit4',where=asNamespace('mgcv'))
.captured$sp[1:2]<-c(-.44054612,-2.60251993)
result<-lapply(c(0,2),function(k){
 aa<-.captured;aa$deriv<-k
 b<-do.call(mgcv:::gam.fit4,aa)
 list(deriv=k,coef=b$coefficients,score=b$REML,gradient=b$REML1,hessian=b$REML2,
 converged=b$converged,iter=b$iter,dev=b$deviance,scale=b$scale.est,weights=b$weights,warn=b$warn)
})
write_json(list(version=as.character(packageVersion('mgcv')),sp=.captured$sp,result=result),a[2],digits=17,auto_unbox=TRUE,na='null')
