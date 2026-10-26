suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1],simplifyVector=FALSE)
d<-as.data.frame(lapply(p$data,unlist));d$y<-as.integer(d$y_poisson>1);d$B<-I(as.matrix(d[,c('x','u','v')]))
P<-lapply(p$penalties,function(S)do.call(rbind,lapply(S,unlist)))
P$L<-do.call(rbind,lapply(p$L,unlist));P$sp<-unlist(p$sp)
probe<-function(expr)tryCatch({b<-expr;list(ok=TRUE,sp=b$sp)},error=function(e)list(ok=FALSE,error=conditionMessage(e),call=paste(deparse(conditionCall(e)),collapse=' ')))
cases<-list(general_list=probe(gam(list(y~B),data=d,family=multinom(K=1),paraPen=list(B=P),method='REML')),
 general_flat=probe(gam(y~B,data=d,family=multinom(K=1),paraPen=list(B=P),method='REML')))
if(requireNamespace('scam',quiet=TRUE)) {
 d$y<-d$y_gaussian
 cases$transformed<-probe(scam::scam(y~B+s(z,bs='mpi',k=5),data=d,paraPen=list(B=P)))
}
write_json(list(version=as.character(packageVersion('mgcv')),cases=cases),a[2],digits=17,auto_unbox=TRUE)
