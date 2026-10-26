suppressPackageStartupMessages(library(mgcv));library(jsonlite)
stopifnot(as.character(packageVersion('mgcv'))=='1.9.4')
set.seed(615);n<-96;q<-3;nt<-21
X<-matrix(runif(n*q,-1,1),n,q);Z<-matrix(runif(n*q,-1,1),n,q)
L<-matrix(runif(n*q,.1,.9),n,q);L[4,]<-0
NX<-matrix(runif(nt*q,-1,1),nt,q);NZ<-matrix(runif(nt*q,-1,1),nt,q);NL<-matrix(runif(nt*q,.1,.9),nt,q);NL[2,]<-0
f<-factor(rep(c('c','a','b'),length.out=n),levels=c('c','a','b'));nf<-factor(rep(c('b','c','a'),length.out=nt),levels=levels(f))
noise<-rnorm(n,sd=.07);cases<-list()
for(bs in c('tp','cr','ps'))for(kind in c('unit','varying','constant'))for(method in c('fixed','REML'))for(dims in c(1,2))for(repeated in c(FALSE,TRUE)) {
 if(dims==2&&bs!='tp')next
 if(repeated&&bs=='tp')next
 XX<-if(repeated)matrix(rep(seq(-1,1,length.out=11),length.out=n*q),n,q) else X
 W<-L;NW<-NL
 if(kind=='unit'){W[,]<-1;NW[,]<-1}
 if(kind=='constant'){W[4,]<-1;W<-W/rowSums(W);NW[2,]<-1;NW<-NW/rowSums(NW)}
 data<-list(X=XX,Z=Z,L=W,f=f,response=.4+rowSums(W*(sin(XX)+if(dims==2).2*Z else 0))+.15*as.integer(f)+noise)
 nd<-list(X=NX,Z=NZ,L=NW,f=nf)
 expr<-sprintf("s(%s,f,bs='fs',k=6,xt=list(bs='%s')%s)",if(dims==1)'X' else 'X,Z',bs,if(kind=='unit')'' else ',by=L')
 sm<-smoothCon(eval(parse(text=expr)),data,n=n,absorb.cons=TRUE)[[1]]
 sp<-rep(.3,length(sm$S));form<-as.formula(paste('response~',expr))
 b<-if(method=='fixed')gam(form,data=data,sp=sp,method='REML') else gam(form,data=data,method='REML')
 pr<-predict(b,nd,se.fit=TRUE);P<-predict(b,nd,type='lpmatrix')
 cases[[paste0(paste(bs,kind,method,dims,sep='/'),if(repeated)'/repeated' else '')]]<-list(data=data,newdata=nd,levels=levels(f),sp=unname(if(method=='fixed')sp else b$sp),
 X=unname(predict(b,type='lpmatrix')),Xp=unname(P),S=unname(lapply(b$smooth[[1]]$S,unname)),fit=unname(pr$fit),se=unname(pr$se.fit),fitted=unname(fitted(b)),edf=sum(b$edf),covariance=unname(P%*%b$Vp%*%t(P)))
}
# Source-owned composition boundaries for ordinary soap base smooths.
d<-data.frame(x=X[,1]*.8,y=Z[,1]*.8,f=f,response=data$response)
bnd<-list(list(x=c(-1,1,1,-1,-1),y=c(-1,-1,1,1,-1)))
kn<-list(x=c(-.5,.5,-.5,.5),y=c(-.5,-.5,.5,.5));nesting<-list()
for(mode in c('fs','sz'))for(bs in c('so','sf','sw','ad')) {
 expr<-sprintf("s(x,y,f,bs='%s',k=%d,xt=list(bs='%s',bnd=bnd,nmax=30))",mode,if(bs=='ad')16 else 8,bs)
 if(bs=='ad')expr<-sprintf("s(x,f,bs='%s',k=12,xt=list(bs='ad'))",mode)
 nesting[[paste(mode,bs,sep='/')]]<-tryCatch({sm<-smoothCon(eval(parse(text=expr)),d,knots=if(bs=='ad')NULL else kn,absorb.cons=TRUE)[[1]];list(accepted=TRUE,columns=ncol(sm$X),penalties=length(sm$S))},error=function(e)list(accepted=FALSE,error=conditionMessage(e)))
}
write_json(list(version=as.character(packageVersion('mgcv')),cases=cases,nesting=nesting),commandArgs(TRUE)[1],digits=17,auto_unbox=TRUE)
