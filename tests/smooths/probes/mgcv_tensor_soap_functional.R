suppressPackageStartupMessages(library(mgcv));library(jsonlite)
stopifnot(as.character(packageVersion('mgcv'))=='1.9.4')
set.seed(312);n<-110;q<-3;nt<-23
x<-matrix(runif(n*q,-.85,.85),n,q);y<-matrix(runif(n*q,-.85,.85),n,q);z<-matrix(runif(n*q,-1,1),n,q)
L<-matrix(runif(n*q,.1,1),n,q);o<-seq(-.2,.15,length.out=n)
d<-list(x=x,y=y,z=z,L=L,o=o,response=.3+rowSums(L*(sin(x)+.3*cos(y)+.4*z+.2*x*z))+o+rnorm(n,sd=.06))
nd<-list(x=matrix(runif(nt*q,-.8,.8),nt,q),y=matrix(runif(nt*q,-.8,.8),nt,q),z=matrix(runif(nt*q,-.9,.9),nt,q),L=matrix(runif(nt*q,.1,1),nt,q),o=seq(.1,-.1,length.out=nt))
bnd<-list(list(x=c(-1,1,1,-1,-1),y=c(-1,-1,1,1,-1)))
kn<-list(x=c(-.5,.5,-.5,.5),y=c(-.5,-.5,.5,.5))
ans<-list()
for(fun in c('te','ti','t2'))for(bs in c('sf','sw'))for(method in c('fixed','REML')) {
 expr<-sprintf("%s(x,y,z,by=L,bs=c('%s','cr'),d=c(2,1),k=c(8,4),xt=list(list(bnd=bnd,nmax=30),NULL))",fun,bs)
 sm<-smoothCon(eval(parse(text=expr)),d,n=n,knots=kn,absorb.cons=TRUE)[[1]]
 form<-as.formula(paste('response~',expr,'+offset(o)'))
 sp<-rep(.3,length(sm$S))
 b<-if(method=='fixed')gam(form,data=d,knots=kn,sp=sp,method='REML') else gam(form,data=d,knots=kn,method='REML')
 pr<-predict(b,nd,se.fit=TRUE);P<-predict(b,nd,type='lpmatrix')
 ans[[paste(fun,bs,method,sep='/')]]<-list(data=d,newdata=nd,bnd=bnd,knots=kn,sp=if(method=='fixed')sp else unname(b$sp),
  X=unname(predict(b,type='lpmatrix')),Xp=unname(P),coef=unname(coef(b)),S=unname(lapply(b$smooth[[1]]$S,unname)),
  fit=unname(pr$fit),se=unname(pr$se.fit),fitted=unname(fitted(b)),edf=sum(b$edf),covariance=unname(P%*%b$Vp%*%t(P)),
  raw_X=unname(sm$X),raw_S=unname(lapply(sm$S,unname)),null_space_dim=sm$null.space.dim,rank=sm$rank,converged=b$converged)
}
write_json(list(version=as.character(packageVersion('mgcv')),cases=ans),commandArgs(TRUE)[1],digits=17,auto_unbox=TRUE)
