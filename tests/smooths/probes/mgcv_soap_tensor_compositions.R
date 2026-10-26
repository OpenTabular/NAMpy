suppressPackageStartupMessages(library(mgcv));library(jsonlite)
stopifnot(as.character(packageVersion('mgcv'))=='1.9.4')
a<-commandArgs(TRUE);base<-fromJSON(a[1],simplifyDataFrame=FALSE);ans<-list()
for(fun in c('te','ti','t2'))for(geometry in c('free','hole','known','mixed','known_hole'))for(bs in c('so','sf')) {
 d<-do.call(rbind,lapply(base$data,as.data.frame));nd<-do.call(rbind,lapply(base$newdata,as.data.frame));bnd<-base$bnd
 if(geometry %in% c('hole','mixed','known_hole')) {
  bnd[[2]]<-list(x=c(-.2,.2,.2,-.2,-.2),y=c(-.2,-.2,.2,.2,-.2))
  d<-d[abs(d$x)>.25|abs(d$y)>.25,];nd<-nd[abs(nd$x)>.25|abs(nd$y)>.25,]
 }
 if(geometry %in% c('known','mixed','known_hole'))bnd[[1]]$f<-c(0,.3,.4,.1,0)
 if(geometry=='known_hole')bnd[[2]]$f<-c(.2,.1,0,.1,.2)
 expr<-sprintf("%s(x,y,z,bs=c('%s','cr'),d=c(2,1),k=c(8,4),xt=list(list(bnd=bnd,nmax=30),NULL))",fun,bs)
 marginal<-tryCatch(smooth.construct(s(x,y,bs=bs,k=8,xt=list(bnd=bnd,nmax=30)),d,base$knots),error=function(e)e)
 result<-tryCatch({
  sm<-smoothCon(eval(parse(text=expr)),d,knots=base$knots,absorb.cons=TRUE)[[1]]
  list(accepted=TRUE,npen=length(sm$S),ncoef=ncol(sm$X),marginal_npen=length(marginal$S),marginal_offset=if(is.null(attr(marginal$X,'offset')))0 else max(abs(attr(marginal$X,'offset'))),tensor_offset=if(is.null(attr(sm$X,'offset')))0 else max(abs(attr(sm$X,'offset'))))
 },error=function(e)list(accepted=FALSE,error=conditionMessage(e),marginal_npen=if(inherits(marginal,'error'))NA else length(marginal$S)))
 result$data<-d;result$newdata<-nd;result$bnd<-bnd;result$knots<-base$knots
 if(result$accepted) {
  result$fits<-list()
  for(method in c('fixed','REML')) {
   form<-as.formula(paste('response~',expr,'+offset(o)'))
   fit<-tryCatch({
    b<-gam(form,data=d,knots=base$knots,sp=if(method=='fixed')rep(.3,result$npen) else NULL,method='REML')
    pr<-predict(b,nd,se.fit=TRUE);P<-predict(b,nd,type='lpmatrix')
    list(accepted=TRUE,sp=if(method=='fixed')rep(.3,result$npen) else unname(b$sp),fit=unname(pr$fit),se=unname(pr$se.fit),train_fit=unname(fitted(b)),edf=sum(b$edf),score=unname(b$gcv.ubre),X=unname(model.matrix(b)),P=unname(P),fit_X=unname(sm$X),S=unname(lapply(sm$S,unname)),covariance=unname(P%*%b$Vp%*%t(P)))
   },error=function(e)list(accepted=FALSE,error=conditionMessage(e)))
   result$fits[[method]]<-fit
  }
 }
 ans[[paste(fun,bs,geometry,sep='/')]]<-result
}
write_json(list(version=as.character(packageVersion('mgcv')),R=as.character(getRversion()),cases=ans),a[2],digits=17,auto_unbox=TRUE)
