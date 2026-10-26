# Why default smoothCon X and PredictMat differ for scalar t2(sf,cr).
suppressPackageStartupMessages(library(mgcv))
set.seed(231);n<-48;q<-3
X<-matrix(runif(n*q,-.75,.75),n,q);Y<-matrix(runif(n*q,-.75,.75),n,q);Z<-matrix(runif(n*q,-.75,.75),n,q)
d<-data.frame(X=X[,1],Y=Y[,1],Z=Z[,1])
bnd<-list(list(X=c(-1,1,1,-1,-1),Y=c(-1,-1,1,1,-1)))
kn<-list(X=c(-.5,.5,-.5,.5),Y=c(-.5,-.5,.5,.5))
sp<-t2(X,Y,Z,bs=c('sf','cr'),d=c(2,1),k=c(8,4),xt=list(list(bnd=bnd,nmax=30),NULL))
for(mc in 0:2) {
 sm<-smoothCon(sp,d,knots=kn,absorb.cons=TRUE,modCon=mc)[[1]]
 cat('modCon',mc,'constructor prediction difference',max(abs(sm$X-PredictMat(sm,d))),'\n')
}
# A fitted gam carries the alternative prediction parameterization correctly.
d$y<-sin(d$X)+d$Z+rnorm(n,sd=.1)
b<-gam(y~t2(X,Y,Z,bs=c('sf','cr'),d=c(2,1),k=c(8,4),xt=list(list(bnd=bnd,nmax=30),NULL)),data=d,knots=kn,sp=c(.1,.1,.1),method='REML')
cat('gam fitted prediction difference',max(abs(fitted(b)-predict(b))), '\n')
