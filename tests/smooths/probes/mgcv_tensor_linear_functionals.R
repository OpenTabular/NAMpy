# Reference-only fit/predict evidence for the first matrix-functional slice.
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
stopifnot(as.character(packageVersion('mgcv'))=='1.9.4')
set.seed(614);n<-90;q<-3;nt<-19
X<-matrix(runif(n*q,-1.4,1.8),n,q);Z<-matrix(runif(n*q,-1,1),n,q)
L<-matrix(runif(n*q,.1,1),n,q);L[seq(4,n,11),2]<-0
NX<-matrix(runif(nt*q,-1.5,1.9),nt,q);NZ<-matrix(runif(nt*q,-1.1,1.1),nt,q)
NL<-matrix(runif(nt*q,.1,1),nt,q);NL[3,]<-0
w<-runif(n,.5,1.8);w[c(7,23)]<-0
o<-seq(-.15,.25,length.out=n);no<-seq(.1,-.2,length.out=nt)
noise<-rnorm(n,sd=.12);cases<-list();X0<-X;Z0<-Z
for(tensor in c('te','ti','t2')) for(bs in c('cr','ps','bs','tp','ds','gp')) for(method in c('fixed','REML')) for(kind in c('unweighted','weighted','constant')) for(dimensions in 2) for(repeated in c(FALSE,TRUE)) if(!repeated || bs=='cr') {
  X<-X0;Z<-Z0
  if(repeated) { X[,2:3]<-rep(X[1:30,1],length.out=n*2);Z[,2:3]<-rep(Z[1:30,1],length.out=n*2) }
  weighted<-kind!='unweighted'
  WL<-if(kind=='constant') L/rowSums(L) else L
  WNL<-if(kind=='constant') {tmp<-NL;tmp[3,]<-1;tmp/rowSums(tmp)} else NL
  W<-if(weighted)WL else matrix(1,n,q)
  y<-.4+o+rowSums(W*(sin(X)+if(dimensions==2) .3*Z^2+.2*sin(2*X*Z) else 0))+noise
  data<-list(y=y,X=X,Z=Z,L=WL,o=o,w=w)
  nd<-list(X=NX,Z=NZ,L=WNL,o=no)
  form<-as.formula(sprintf("y~%s(X,Z,bs=c('%s','%s'),k=c(5,5)%s)+offset(o)",tensor,bs,bs,if(weighted)',by=L' else ''))
  nsp<-if(tensor=='t2') 3 else 2
  fixed_sp<-seq(.4,.8,length.out=nsp)
  fit<-if(method=='fixed')gam(form,data=data,weights=w,sp=fixed_sp,method='REML') else gam(form,data=data,weights=w,method='REML')
  A<-predict(fit,type='lpmatrix');B<-predict(fit,nd,type='lpmatrix')
  pr<-predict(fit,nd,se.fit=TRUE); tr<-predict(fit,se.fit=TRUE)
  # Predict individual quadrature locations, then sum weighted smooth columns.
  point<-list(X=as.vector(NX),Z=as.vector(NZ),L=rep(1,nt*q),o=rep(0,nt*q))
  P<-predict(fit,point,type='lpmatrix');aggregate<-matrix(0,nt,ncol(P));aggregate[,1]<-1
  NW<-if(weighted)WNL else matrix(1,nt,q)
  for(j in 1:q)aggregate[,-1]<-aggregate[,-1]+P[((j-1)*nt+1):(j*nt),-1]*NW[,j]
  key<-paste(tensor,bs,method,kind,sep='/')
  if(repeated) key<-paste0(key,'/repeated')
  raw<-smoothCon(interpret.gam(form)$smooth.spec[[1]],data,n=n,absorb.cons=FALSE)[[1]]
  constrained<-smoothCon(interpret.gam(form)$smooth.spec[[1]],data,n=n,absorb.cons=TRUE)[[1]]
  cases[[key]]<-list(tensor=tensor,basis=bs,method=method,weighted=weighted,dimensions=dimensions,formula=deparse(form),
    data=data,newdata=nd,coef=unname(coef(fit)),Vp=unname(fit$Vp),sp=unname(if(method=='fixed') fixed_sp else fit$sp),
    scale=fit$sig2,edf=unname(fit$edf),X=unname(A),Xp=unname(B),S=lapply(fit$smooth[[1]]$S,unname),
    train_fit=unname(tr$fit),train_se=unname(tr$se.fit),fit=unname(pr$fit),se=unname(pr$se.fit),
    covariance_kernel=unname(B%*%fit$Vp%*%t(B)),point_aggregate=unname(aggregate),
    fit_X=unname(constrained$X),fit_S=lapply(constrained$S,unname),
    raw_X=unname(raw$X),raw_C=unname(raw$C),raw_S=lapply(raw$S,unname),
    smooth_rank=fit$smooth[[1]]$rank,null_space_dim=fit$smooth[[1]]$null.space.dim,
    summation_error=max(abs(B-aggregate)),converged=fit$converged)
}
write_json(list(version=as.character(packageVersion('mgcv')),seed=614,cases=cases),commandArgs(TRUE)[1],digits=17,auto_unbox=TRUE)
