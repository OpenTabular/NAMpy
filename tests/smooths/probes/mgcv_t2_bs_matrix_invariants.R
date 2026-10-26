# Actual mgcv reference for supported matrix t2 B-spline invariants.
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1]);d<-p$data;nd<-p$newdata
stopifnot(as.character(packageVersion('mgcv'))=='1.9.4')
by<-if(p$kind=='unweighted')'' else ',by=L'
opt<-if(p$route=='range_only')',ord=2' else if(p$route=='one_null')',m=list(c(3,1),c(3,1))' else ',m=list(c(3,0),c(3,0))'
form<-as.formula(sprintf("y~t2(X,Z,bs=c('bs','bs'),k=c(5,5),full=%s%s%s)+offset(o)",p$full,by,opt))
nsp<-if(p$route=='one_null')3 else 1
b<-gam(form,data=d,weights=d$w,sp=if(p$method=='fixed')seq(.4,.8,length.out=nsp) else NULL,method='REML')
P<-predict(b,nd,type='lpmatrix');pr<-predict(b,nd,se.fit=TRUE)
constrained<-smoothCon(interpret.gam(form)$smooth.spec[[1]],d,n=length(d$y),absorb.cons=TRUE)[[1]]
ans<-list(version=as.character(packageVersion('mgcv')),converged=b$converged,
coef=unname(coef(b)),Vp=unname(b$Vp),edf=sum(b$edf),score=unname(b$gcv.ubre),scale=b$sig2,
sp=unname(if(p$method=='fixed')seq(.4,.8,length.out=nsp) else b$sp),
train_X=unname(model.matrix(b)),new_X=unname(P),penalties=lapply(b$smooth[[1]]$S,unname),
fit_X=unname(constrained$X),
train_fit=unname(fitted(b)),fit=unname(pr$fit),se=unname(pr$se.fit),covariance=unname(P%*%b$Vp%*%t(P)))
write_json(ans,a[2],digits=17,auto_unbox=TRUE)
