# Explicit factor+dim input; no classless array is silently promoted to a factor.
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1],simplifyVector=TRUE)
d<-p$data;nd<-p$newdata;lev<-p$levels
factor_matrix<-function(x) structure(matrix(as.integer(factor(as.vector(x),levels=lev)),nrow(x)),levels=lev,class='factor')
d$f<-factor_matrix(d$f);nd$f<-factor_matrix(nd$f)
if(!is.null(d$g)) {d$g<-factor_matrix(d$g);nd$g<-factor_matrix(nd$g)}
soap<-p$basis %in% c('so','sf','sw')
raw<-fromJSON(a[1],simplifyVector=FALSE)
bnd<-lapply(raw$boundary,function(b) lapply(b,unlist));kn<-if(soap)p$knots else NULL
xt<-if(soap)sprintf("list(bs='%s',bnd=bnd,nmax=30)",p$basis) else sprintf("list(bs='%s')",p$basis)
by<-if(p$kind=='unit') '' else ',by=L'
expr<-sprintf("s(%s,f%s,bs='%s',k=8,xt=%s%s%s)",if(soap)'X,Y' else 'X',if(is.null(d$g))'' else ',g',p$mode,xt,by,if(p$basis=='tp')',m=1' else if(p$basis=='ps' && is.null(p$default_ps_order))',m=c(2,1)' else '')
form<-as.formula(paste('response~',expr,'+offset(o)'))
warnings<-character()
ans<-tryCatch(withCallingHandlers({
 sm<-smoothCon(eval(parse(text=expr)),d,n=length(d$response),knots=kn,absorb.cons=TRUE)[[1]]
 sp<-if(is.null(p$fixed_sp))rep(.3,length(sm$S)) else p$fixed_sp
 b<-gam(form,data=d,knots=kn,weights=d$w,sp=if(p$method=='fixed')sp else NULL,method='REML')
 pr<-predict(b,nd,se.fit=TRUE);P<-predict(b,nd,type='lpmatrix')
 list(accepted=TRUE,sp=unname(if(p$method=='fixed')sp else b$sp),X=unname(predict(b,type='lpmatrix')),Xp=unname(P),S=unname(lapply(b$smooth[[1]]$S,unname)),fit=unname(pr$fit),se=unname(pr$se.fit),fitted=unname(fitted(b)),edf=sum(b$edf),covariance=unname(P%*%b$Vp%*%t(P)),factor_offset=attr(sm$X,'offset'),term_prediction_offset=attr(PredictMat(b$smooth[[1]],nd,n=length(nd$o)),'offset'))
},warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e)list(accepted=FALSE,error=conditionMessage(e)))
ans$warnings<-warnings;ans$version<-as.character(packageVersion('mgcv'))
write_json(ans,a[2],digits=17,auto_unbox=TRUE,na='null',null='null')
