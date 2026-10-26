suppressPackageStartupMessages(library(mgcv));library(jsonlite)
stopifnot(as.character(packageVersion('mgcv'))=='1.9.4')
set.seed(718);n<-90;q<-3;nt<-17
X<-matrix(runif(n*q,-1,1),n,q);Z<-matrix(runif(n*q,.2,1.4),n,q)
F<-factor(sample(letters[1:6],n*q,TRUE),levels=letters[1:6]);dim(F)<-c(n,q)
G<-factor(sample(LETTERS[1:3],n*q,TRUE),levels=LETTERS[1:3]);dim(G)<-c(n,q)
L<-matrix(runif(n*q,.1,1),n,q);L[seq(4,n,11),2]<-0
NX<-matrix(runif(nt*q,-1,1),nt,q);NZ<-matrix(runif(nt*q,.2,1.4),nt,q)
NF<-factor(sample(letters[1:6],nt*q,TRUE),levels=letters[1:6]);dim(NF)<-c(nt,q)
NG<-factor(sample(LETTERS[1:3],nt*q,TRUE),levels=LETTERS[1:3]);dim(NG)<-c(nt,q)
NL<-matrix(runif(nt*q,.1,1),nt,q);NL[3,]<-0
nb<-list(a='b',b=c('a','c'),c=c('b','d'),d=c('c','e'),e=c('d','f'),f='e')
w<-runif(n,.5,1.8);w[c(7,23)]<-0;o<-seq(-.15,.25,length.out=n);no<-seq(.1,-.2,length.out=nt)
noise<-rnorm(n,sd=.12);cases<-list()
for(kind in c('re_numeric','re_numeric2','re_factor','re_mixed','re_factor2','mrf_full','mrf_low')) for(method in c('fixed','REML')) for(weighting in c('unweighted','weighted','constant')) {
 weighted<-weighting!='unweighted';WL<-if(weighting=='constant')L/rowSums(L) else L
 WNL<-if(weighting=='constant'){tmp<-NL;tmp[3,]<-1;tmp/rowSums(tmp)}else NL
 W<-if(weighted)WL else matrix(1,n,q)
 surface<-switch(kind,re_numeric=.6*X,re_numeric2=.6*X*Z,re_factor=matrix(as.numeric(F),n,q)/3,re_mixed=X*matrix(as.numeric(F),n,q)/3,re_factor2=matrix(as.numeric(F),n,q)*matrix(as.numeric(G),n,q)/6,matrix(as.numeric(F),n,q)/3)
 y<-.4+o+rowSums(W*surface)+noise
 terms<-switch(kind,re_numeric='X',re_numeric2='X,Z',re_factor='F',re_mixed='X,F',re_factor2='F,G','F')
 extra<-if(grepl('mrf',kind))sprintf(",bs='mrf',k=%s,xt=list(nb=nb)",if(kind=='mrf_low')'4' else '-1') else ",bs='re'"
 form<-as.formula(sprintf('y~s(%s%s%s)+offset(o)',terms,extra,if(weighted)',by=L' else ''))
 data<-list(y=y,X=X,Z=Z,F=F,G=G,L=WL,w=w,o=o);nd<-list(X=NX,Z=NZ,F=NF,G=NG,L=WNL,o=no)
 fit<-if(method=='fixed')gam(form,data=data,weights=w,sp=.4,method='REML') else gam(form,data=data,weights=w,method='REML')
 A<-predict(fit,type='lpmatrix');B<-predict(fit,nd,type='lpmatrix');pr<-predict(fit,nd,se.fit=TRUE)
 point<-list(X=as.vector(NX),Z=as.vector(NZ),F=factor(as.vector(NF),levels=letters[1:6]),G=factor(as.vector(NG),levels=LETTERS[1:3]),L=rep(1,nt*q),o=rep(0,nt*q))
 P<-predict(fit,point,type='lpmatrix');agg<-matrix(0,nt,ncol(P));agg[,1]<-1;NW<-if(weighted)WNL else matrix(1,nt,q)
 for(j in 1:q)agg[,-1]<-agg[,-1]+P[((j-1)*nt+1):(j*nt),-1,drop=FALSE]*NW[,j]
 # jsonlite handles factor matrices as flattened vectors, so serialize their labels explicitly.
 data$F<-matrix(as.character(F),n,q);data$G<-matrix(as.character(G),n,q)
 nd$F<-matrix(as.character(NF),nt,q);nd$G<-matrix(as.character(NG),nt,q)
 cases[[paste(kind,method,weighting,sep='/')]]<-list(data=data,newdata=nd,coef=unname(coef(fit)),X=A,Xp=B,S=lapply(fit$smooth[[1]]$S,unname),Vp=unname(fit$Vp),edf=sum(fit$edf),sp=unname(if(method=='fixed').4 else fit$sp),fitted=unname(fitted(fit)),fit=unname(pr$fit),se=unname(pr$se.fit),covariance_kernel=unname(B%*%fit$Vp%*%t(B)),point_aggregate=agg,summation_error=max(abs(B-agg)),converged=fit$converged)
}
write_json(list(version=as.character(packageVersion('mgcv')),cases=cases),commandArgs(TRUE)[1],digits=17,auto_unbox=TRUE)
