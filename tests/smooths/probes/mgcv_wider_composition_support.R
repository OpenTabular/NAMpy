# Purpose-built upstream acceptance probe; never used by ordinary test runs.
suppressPackageStartupMessages(library(mgcv))
library(jsonlite)
stopifnot(as.character(packageVersion('mgcv')) == '1.9.4')
set.seed(231)
n<-48L; q<-3L
X<-matrix(runif(n*q,-.75,.75),n,q);Y<-matrix(runif(n*q,-.75,.75),n,q)
Z<-matrix(runif(n*q,-.75,.75),n,q);L<-matrix(runif(n*q,.2,.9),n,q)
f<-factor(rep(letters[1:3],length.out=n));g<-factor(rep(LETTERS[1:2],each=n/2))
d<-list(X=X,Y=Y,Z=Z,L=L,x=X[,1],y=Y[,1],z=Z[,1],f=f,g=g)
bnd<-list(list(X=c(-1,1,1,-1,-1),Y=c(-1,-1,1,1,-1)))
kn<-list(X=c(-.5,.5,-.5,.5),Y=c(-.5,-.5,.5,.5))
ans<-list()
probe<-function(label,expr,data=d,knots=NULL) {
  warns<-character(); ans[[label]]<<-tryCatch(withCallingHandlers({
    spec<-eval(parse(text=expr)); sm<-smoothCon(spec,data=data,knots=knots,n=n,absorb.cons=TRUE)
    pp<-lapply(sm,function(s) PredictMat(s,data,n=n))
    list(accepted=TRUE,expression=expr,dim=lapply(sm,function(s) dim(s$X)),
         prediction_dim=lapply(pp,dim),prediction_error=lapply(seq_along(sm),function(i) max(abs(pp[[i]]-sm[[i]]$X))),
         penalties=lapply(sm,function(s) length(s$S)),warnings=warns)
  },warning=function(w){warns<<-c(warns,conditionMessage(w));invokeRestart('muffleWarning')}),
  error=function(e) list(accepted=FALSE,expression=expr,error=conditionMessage(e),warnings=warns))
}
for(bs in c('tp','ts','cr','cs','cc','ps','cp','bs','ds','gp','re','ad')) {
  for(by in c(FALSE,TRUE)) probe(paste('matrix',bs,by,sep='/'),sprintf("s(X,bs='%s',k=8%s)",bs,if(by) ',by=L' else ''))
}
probe('matrix/sos',"s(X,Y,bs='sos',k=8,by=L)")
for(bs in c('so','sf','sw')) probe(paste('matrix',bs,sep='/'),sprintf("s(X,Y,bs='%s',k=8,xt=list(bnd=bnd,nmax=30),by=L)",bs),knots=kn)
for(fun in c('te','ti','t2')) for(bs in c('tp','cr','ps','bs','ds','gp','re','ad'))
 probe(paste('matrix_tensor',fun,bs,sep='/'),sprintf("%s(X,Y,bs=c('%s','cr'),k=c(5,5),by=L)",fun,bs))
probe('matrix/factor_by',"s(X,bs='cr',k=8,by=f)")
for(bs in c('fs','sz')) {
 probe(paste('matrix_factor',bs,sep='/'),sprintf("s(X,f,bs='%s',k=5,by=L)",bs))
 for(base in c('cr','ps','tp','ds','gp','ad')) probe(paste('factor_base',bs,base,sep='/'),sprintf("s(x,f,bs='%s',k=5,xt=list(bs='%s'))",bs,base))
 probe(paste('nested_factor',bs,sep='/'),sprintf("s(x,f,g,bs='%s',k=5)",bs))
 for(fun in c('te','ti','t2')) probe(paste('factor_tensor',fun,bs,sep='/'),sprintf("%s(x,f,z,bs=c('%s','cr'),d=c(2,1),k=c(5,4))",fun,bs))
}
for(fun in c('te','ti','t2')) for(bs in c('so','sf','sw'))
 probe(paste('soap_tensor',fun,bs,sep='/'),sprintf("%s(X,Y,Z,bs=c('%s','cr'),d=c(2,1),k=c(8,4),xt=list(list(bnd=bnd,nmax=30),NULL),by=L)",fun,bs),knots=kn)
# Distinguish valid larger adaptive bases from insufficient k in probes above.
for(fun in c('te','ti','t2')) probe(paste('adaptive_tensor',fun,sep='/'),sprintf("%s(X,Y,bs=c('ad','cr'),k=c(12,5),by=L)",fun))
for(bs in c('fs','sz')) probe(paste('adaptive_factor',bs,sep='/'),sprintf("s(x,f,bs='%s',k=12,xt=list(bs='ad'))",bs))
# Soap tensor margins also work for ordinary scalar-valued covariates.
dscalar<-d;dscalar$X<-X[,1];dscalar$Y<-Y[,1];dscalar$Z<-Z[,1];dscalar$L<-L[,1]
for(fun in c('te','ti','t2')) for(bs in c('so','sf','sw'))
 probe(paste('scalar_soap_tensor',fun,bs,sep='/'),sprintf("%s(X,Y,Z,bs=c('%s','cr'),d=c(2,1),k=c(8,4),xt=list(list(bnd=bnd,nmax=30),NULL))",fun,bs),data=dscalar,knots=kn)
# Factor-valued matrix inputs retain R factor levels with explicit dimensions.
F<-factor(rep(letters[1:4],length.out=n*q));dim(F)<-c(n,q)
dmrf<-d;dmrf$F<-F
nb<-list(a=c(2L,4L),b=c(1L,3L),c=c(2L,4L),d=c(1L,3L))
probe('matrix/mrf',"s(F,bs='mrf',xt=list(nb=nb),by=L)",data=dmrf)
probe('matrix/factor_re',"s(F,bs='re',by=L)",data=dmrf)
write_json(list(version=as.character(packageVersion('mgcv')),cases=ans),commandArgs(TRUE)[1],digits=17,auto_unbox=TRUE,pretty=TRUE)
