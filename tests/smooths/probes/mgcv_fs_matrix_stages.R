# Diagnose fs matrix natural parameterization; no changes to reference defaults.
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
set.seed(615);n<-96;q<-3
X<-matrix(runif(n*q,-1,1),n,q);Z<-matrix(runif(n*q,-1,1),n,q)
f<-factor(rep(c('c','a','b'),length.out=n),levels=c('c','a','b'))
d<-list(X=X,f=f)
a<-mgcv:::smooth.construct3(s(X,f,bs='fs',k=6),d,NULL)
b<-mgcv:::smooth.construct3(s(X,bs='tp',k=6),d,NULL)
p<-list(X=as.vector(X),f=rep(f,q))
c<-mgcv:::smooth.construct3(s(X,f,bs='fs',k=6),p,NULL)
Q<-qr(b$X);R<-qr.R(Q);RSR<-forwardsolve(t(R),t(forwardsolve(t(R),t(b$S[[1]]))))
write_json(list(data=d,base_X=unname(b$X),base_S=unname(b$S[[1]]),X=unname(a$X),P=unname(a$P),
 R=unname(R),RSR=unname(RSR),packed_index=a$ind,matrix_scalar_error=max(abs(a$X-c$X))),commandArgs(TRUE)[1],digits=17,auto_unbox=TRUE)
