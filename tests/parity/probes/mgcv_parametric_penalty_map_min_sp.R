suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1],simplifyVector=FALSE)
d<-as.data.frame(lapply(p$data,unlist));d$y<-d$y_gaussian;d$B<-I(as.matrix(d[,c('x','u','v')]))
P<-lapply(p$penalties,function(S)do.call(rbind,lapply(S,unlist)))
P$L<-do.call(rbind,lapply(p$L,unlist));P$sp<-unlist(p$sp)
cases<-list()
for(smooth in c(FALSE,TRUE)) {
 form<-if(smooth)y~B+s(z,bs='cr',k=5,sp=.4) else y~B
 cases[[if(smooth)'with_smooth' else 'parametric_only']]<-tryCatch({
  G<-gam(form,data=d,paraPen=list(B=P),method='REML',min.sp=rep(.2,length(P$sp)+as.integer(smooth)),fit=FALSE)
  b<-gam(form,data=d,paraPen=list(B=P),method='REML',min.sp=rep(.2,length(P$sp)+as.integer(smooth)))
  list(ok=TRUE,H=G$H,sp=b$sp,full_sp=b$full.sp,fitted=unname(fitted(b)))
 },error=function(e)list(ok=FALSE,error=conditionMessage(e)))
}
write_json(cases,a[2],digits=17,auto_unbox=TRUE,null='null')
