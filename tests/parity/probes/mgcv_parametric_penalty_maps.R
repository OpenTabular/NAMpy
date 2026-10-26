suppressPackageStartupMessages(library(mgcv));library(jsonlite)
args<-commandArgs(TRUE);p<-fromJSON(args[1],simplifyVector=FALSE)
d<-as.data.frame(lapply(p$data,unlist));d$y<-d[[paste0('y_',p$family)]]
d$B<-I(as.matrix(d[,c('x','u','v')]))
P<-lapply(p$penalties,function(S)do.call(rbind,lapply(S,unlist)))
P$L<-do.call(rbind,lapply(p$L,unlist));P$sp<-unlist(p$sp)
fam<-switch(p$family,gaussian=gaussian(),poisson=poisson(),nb=nb(),scat=scat())
formula<-if(isTRUE(p$smooth))y~B+s(z,bs='cr',k=5,sp=.4) else y~B
control<-gam.control(epsilon=1e-10,newton=list(conv.tol=1e-10))
result<-tryCatch({
 G<-gam(formula,data=d,family=fam,paraPen=list(B=P),method='REML',fit=FALSE)
 b<-gam(formula,data=d,family=fam,paraPen=list(B=P),method='REML',
        optimizer=if(p$optimizer=='efs')'efs' else c('outer',p$optimizer),control=control)
 nd<-d[seq(2,nrow(d),length.out=13),,drop=FALSE]
 nd$B<-I(as.matrix(nd[,c('x','u','v')]))
 prediction<-predict(b,nd,type='link',se.fit=TRUE);Xp<-predict(b,nd,type='lpmatrix')
 out<-list(ok=TRUE,setup=list(L=G$L,lsp0=G$lsp0,sp=G$sp,S=G$S,off=G$off,rank=G$rank),
  prediction=unname(prediction$fit),se=unname(prediction$se.fit),fitted=unname(predict(b,type='link')),
  X=unname(predict(b,type='lpmatrix')),Xp=unname(Xp),sp=unname(b$sp),full_sp=unname(b$full.sp),
  covariance=unname(Xp%*%b$Vp%*%t(Xp)),freq_covariance=unname(Xp%*%b$Ve%*%t(Xp)),
  unconditional_covariance=if(is.null(b$Vc))NULL else unname(Xp%*%b$Vc%*%t(Xp)),
  edf=sum(b$edf),score=unname(b$gcv.ubre),scale=b$sig2,aic=AIC(b),loglik=as.numeric(logLik(b)),
  theta=if(inherits(b$family,'extended.family'))unname(b$family$getTheta(TRUE)) else NULL)
 if(p$family%in%c('gaussian','poisson')) {
  space<-mgcv:::totalPenaltySpace(G$S,G$H,G$off,ncol(G$X))
  roots<-mgcv:::mini.roots(G$S,G$off,ncol(G$X),G$rank)
  UrS<-lapply(roots,function(root)t(space$Y)%*%root)
  U1<-cbind(space$Y,space$Z)
  fam<-mgcv:::fix.family.link(fam);fam<-mgcv:::fix.family.var(fam);fam<-mgcv:::fix.family.ls(fam)
  fixed<-unlist(p$sp)>=0;ev<-unlist(p$eval_sp)[!fixed]
  if(isTRUE(p$smooth))ev<-c(ev)
  L<-G$L;if(is.null(L))L<-diag(length(G$S))
  rho<-drop(L%*%log(ev)+G$lsp0)
  scale<-if(p$family=='gaussian')-1 else 1
  if(scale<0)rho<-c(rho,log(.7))
  fit<-mgcv:::gam.fit3(x=G$X,y=G$y,sp=rho,Eb=space$E,UrS=UrS,weights=G$w,
    U1=U1,Mp=ncol(space$Z),family=fam,control=control,scale=scale,deriv=2,scoreType='REML')
  map<-L
  if(scale<0)map<-rbind(cbind(map,0),c(rep(0,ncol(map)),1))
  out$derivatives<-list(score=unname(fit$REML),gradient=drop(t(map)%*%fit$REML1),
   hessian=unname(t(map)%*%fit$REML2%*%map),evaluation_free=ev,log_scale=if(scale<0)log(.7) else NULL)
 }
 out
},error=function(e)list(ok=FALSE,error=conditionMessage(e),call=paste(deparse(conditionCall(e)),collapse=' ')))
write_json(result,args[2],digits=17,auto_unbox=TRUE,null='null',na='null')
