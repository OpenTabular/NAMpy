suppressPackageStartupMessages(library(mgcv));library(jsonlite)
stopifnot(as.character(packageVersion('mgcv'))=='1.9.4')
set.seed(711);n<-85;q<-3;nt<-17
X<-matrix(runif(n*q,-.85,.85),n,q);Y<-matrix(runif(n*q,-.85,.85),n,q)
pool<-cbind(runif(21,-.85,.85),runif(21,-.85,.85));ix<-sample(1:21,n*q,replace=TRUE,prob=1:21)
X<-matrix(pool[ix,1],n,q);Y<-matrix(pool[ix,2],n,q)
L<-matrix(runif(n*q,.1,1),n,q);L[3,]<-0
NX<-matrix(runif(nt*q,-.85,.85),nt,q);NY<-matrix(runif(nt*q,-.85,.85),nt,q);NL<-matrix(runif(nt*q,.1,1),nt,q);NL[2,]<-0
kn<-list(X=c(-.5,.5,-.5,.5),Y=c(-.5,-.5,.5,.5));noise<-rnorm(n,sd=.04)
o<-seq(-.15,.2,length.out=n);no<-seq(.1,-.1,length.out=nt)
ans<-list()
for(bs in c('so','sf','sw'))for(kind in c('unit','varying','constant'))for(method in 'fixed')for(known in FALSE) {
 W<-L;NW<-NL
 if(kind=='unit'){W[,]<-1;NW[,]<-1}
 if(kind=='constant'){W[3,]<-1;W<-W/rowSums(W);NW[2,]<-1;NW<-NW/rowSums(NW)}
 bnd<-list(list(X=c(-1,1,1,-1,-1),Y=c(-1,-1,1,1,-1)))
 if(known)bnd[[1]]$f<-c(-1,1,1,-1,-1)
 if(known&&bs=='sf')bnd[[2]]<-list(X=c(-.25,.25,.25,-.25,-.25),Y=c(-.25,-.25,.25,.25,-.25))
 d<-list(X=X,Y=Y,L=W,o=o,response=.3+o+rowSums(W*(sin(X)+.3*cos(Y)))+noise)
 nd<-list(X=NX,Y=NY,L=NW,o=no)
 if(known&&bs=='sf'){d$X[abs(d$X)<.3&abs(d$Y)<.3]<-.6;nd$X[abs(nd$X)<.3&abs(nd$Y)<.3]<-.6}
 expr<-sprintf("s(X,Y,bs='%s',k=8,xt=list(bnd=bnd,nmax=30)%s)",bs,if(kind=='unit')'' else ',by=L')
 ans[[paste(bs,kind,method,known,sep='/')]]<-tryCatch({
 spec<-eval(parse(text=expr));sm<-smoothCon(spec,d,n=n,knots=kn,absorb.cons=TRUE)[[1]]
 form<-as.formula(paste('response~',expr,'+offset(o)'));sp<-rep(.3,length(sm$S))
 b<-if(method=='fixed')gam(form,data=d,knots=kn,sp=sp,method='REML') else gam(form,data=d,knots=kn,method='REML')
 pr<-predict(b,nd,se.fit=TRUE);P<-predict(b,nd,type='lpmatrix')
 list(accepted=TRUE,data=d,newdata=nd,bnd=bnd,knots=kn,
 sp=unname(if(method=='fixed')sp else b$sp),X=unname(predict(b,type='lpmatrix')),Xp=unname(P),S=unname(lapply(b$smooth[[1]]$S,unname)),
 fit=unname(pr$fit),se=unname(pr$se.fit),fitted=unname(fitted(b)),edf=sum(b$edf),covariance=unname(P%*%b$Vp%*%t(P)),
 train_offset=unname(attr(PredictMat(b$smooth[[1]],d,n=n),'offset')),prediction_offset=unname(attr(PredictMat(b$smooth[[1]],nd,n=nt),'offset')))
 },error=function(e)list(accepted=FALSE,error=conditionMessage(e),data=d,bnd=bnd,knots=kn))
}
write_json(list(version=as.character(packageVersion('mgcv')),cases=ans),commandArgs(TRUE)[1],digits=17,auto_unbox=TRUE)
