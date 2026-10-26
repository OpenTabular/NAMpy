# Actual mgcv matrix soap factor compositions and source-owned rejections.
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1],simplifyVector=TRUE)
d<-p$data;nd<-p$newdata
lev<-p$levels;d$f<-factor(d$f,levels=lev);nd$f<-factor(nd$f,levels=lev)
# R factors can carry dimensions independently of their class. Classless
# character/numeric matrices are deliberately tested as different inputs.
d$f<-matrix(rep(as.integer(d$f),ncol(d$X)),nrow(d$X))
nd$f<-matrix(rep(as.integer(nd$f),ncol(nd$X)),nrow(nd$X))
if(p$factor_shape=='factor_varying') {
 d$f[,2]<-d$f[,2]%%length(lev)+1
 nd$f[,2]<-nd$f[,2]%%length(lev)+1
}
if(p$factor_shape %in% c('factor_repeated','factor_varying')) {
 d$f<-structure(d$f,levels=lev,class='factor')
 nd$f<-structure(nd$f,levels=lev,class='factor')
} else if(p$factor_shape=='character') {
 d$f<-matrix(lev[d$f],nrow(d$f))
 nd$f<-matrix(lev[nd$f],nrow(nd$f))
}
raw<-fromJSON(a[1],simplifyVector=FALSE);bnd<-lapply(raw$boundary,function(b) lapply(b,unlist));kn<-p$knots
by<-if(p$kind=='unit') '' else ',by=L'
expr<-sprintf("s(X,Y,f,bs='%s',k=8,xt=list(bs='%s',bnd=bnd,nmax=30)%s)",p$mode,p$basis,by)
form<-as.formula(paste('response~',expr,'+offset(o)'))
warnings<-character()
ans<-tryCatch(withCallingHandlers({
 sm<-smoothCon(eval(parse(text=expr)),d,n=length(d$response),knots=kn,absorb.cons=TRUE)[[1]]
 sp<-rep(.3,length(sm$S))
 b<-gam(form,data=d,knots=kn,weights=d$w,sp=if(p$method=='fixed')sp else NULL,method='REML')
 pr<-predict(b,nd,se.fit=TRUE);P<-predict(b,nd,type='lpmatrix')
 # The parent factor constructor replaces object$X and drops its base's offset.
 point<-list(X=as.vector(d$X),Y=as.vector(d$Y))
 base<-smooth.construct(eval(parse(text=sprintf("s(X,Y,bs='%s',k=8,xt=list(bnd=bnd,nmax=30))",p$basis))),point,kn)
 list(accepted=TRUE,sp=unname(if(p$method=='fixed')sp else b$sp),X=unname(predict(b,type='lpmatrix')),Xp=unname(P),S=unname(lapply(b$smooth[[1]]$S,unname)),fit=unname(pr$fit),se=unname(pr$se.fit),fitted=unname(fitted(b)),edf=sum(b$edf),covariance=unname(P%*%b$Vp%*%t(P)),base_offset=attr(base$X,'offset'),factor_offset=attr(sm$X,'offset'),term_prediction_offset=attr(PredictMat(b$smooth[[1]],nd,n=length(nd$o)),'offset'))
},warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e)list(accepted=FALSE,error=conditionMessage(e)))
ans$warnings<-warnings;ans$version<-as.character(packageVersion('mgcv'))
write_json(ans,a[2],digits=17,auto_unbox=TRUE,na='null',null='null')
