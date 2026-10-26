suppressPackageStartupMessages(library(mgcv));library(jsonlite)
stopifnot(as.character(packageVersion('mgcv'))=='1.9.4')
set.seed(718);n<-150;nt<-27
f<-factor(rep(c('c','a','b'),length.out=n),levels=c('c','a','b'))
d<-data.frame(x=runif(n,-.8,.8),y=runif(n,-.8,.8),f=f,w=runif(n,.2,1.2))
d$response<-.3+sin(d$x)*as.integer(f)/3+.2*d$y+rnorm(n,sd=.1)
nd<-data.frame(x=runif(nt,-.8,.8),y=runif(nt,-.8,.8),f=factor(rep(c('b','c','a'),length.out=nt),levels=levels(f)),w=runif(nt,.2,1.2))
bnd<-list(list(x=c(-1,1,1,-1,-1),y=c(-1,-1,1,1,-1)))
kn<-list(x=c(-.5,.5,-.5,.5),y=c(-.5,-.5,.5,.5));cases<-list()
for(mode in c('fs','sz'))for(bs in c('sf','sw'))for(method in c('fixed','REML'))for(by in c(FALSE,TRUE)) for(domain in c('free','mixed')) {
 if(mode=='fs'&&bs=='sw')next
 bnd<-list(list(x=c(-1,1,1,-1,-1),y=c(-1,-1,1,1,-1)))
 if(domain=='mixed')bnd[[2]]<-list(x=c(-.12,-.12,.12,.12,-.12),y=c(-.12,.12,.12,-.12,-.12),f=rep(.4,5))
 # Keep training and held-out points outside the hole.
 d$x[abs(d$x)<.15&abs(d$y)<.15]<-.2;nd$x[abs(nd$x)<.15&abs(nd$y)<.15]<-.2
 expr<-sprintf("s(x,y,f,bs='%s',k=8,xt=list(bs='%s',bnd=bnd,nmax=30)%s)",mode,bs,if(by)',by=w' else '')
 sm<-smoothCon(eval(parse(text=expr)),d,knots=kn,absorb.cons=TRUE)[[1]]
 sp<-rep(.3,length(sm$S));form<-as.formula(paste('response~',expr))
 b<-if(method=='fixed')gam(form,data=d,knots=kn,sp=sp,method='REML')else gam(form,data=d,knots=kn,method='REML')
 pr<-predict(b,nd,se.fit=TRUE);P<-predict(b,nd,type='lpmatrix')
 cases[[paste(mode,bs,method,by,domain,sep='/')]]<-list(boundary=bnd,data=d,newdata=nd,sp=unname(if(method=='fixed')sp else b$sp),X=unname(predict(b,type='lpmatrix')),Xp=unname(P),S=unname(lapply(b$smooth[[1]]$S,unname)),fit=unname(pr$fit),se=unname(pr$se.fit),fitted=unname(fitted(b)),edf=sum(b$edf),covariance=unname(P%*%b$Vp%*%t(P)))
}
write_json(list(version=as.character(packageVersion('mgcv')),cases=cases,levels=levels(f),boundary=bnd,knots=kn),commandArgs(TRUE)[1],digits=17,auto_unbox=TRUE)
