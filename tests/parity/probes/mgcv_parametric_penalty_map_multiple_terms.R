suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1],simplifyVector=FALSE)
d<-as.data.frame(lapply(p$data,unlist));d$y<-d$y_gaussian;d$B<-I(as.matrix(d[,c('x','u','v')]))
P<-lapply(p$penalties,function(S)do.call(rbind,lapply(S,unlist)))
P$L<-do.call(rbind,lapply(p$L,unlist));P$sp<-unlist(p$sp)
Z<-list(matrix(2,1,1),L=matrix(.5,1,1),sp=p$z_sp)
control<-gam.control(epsilon=1e-10,newton=list(conv.tol=1e-10))
G<-gam(y~B+z,data=d,paraPen=list(B=P,z=Z),method='REML',fit=FALSE)
b<-gam(y~B+z,data=d,paraPen=list(B=P,z=Z),method='REML',control=control)
nd<-d[seq(2,nrow(d),length.out=13),,drop=FALSE];nd$B<-I(as.matrix(nd[,c('x','u','v')]))
pr<-predict(b,nd,type='link',se.fit=TRUE);Xp<-predict(b,nd,type='lpmatrix')
write_json(list(ok=TRUE,setup=list(L=G$L,lsp0=G$lsp0,sp=G$sp,S=G$S,off=G$off,rank=G$rank),
 prediction=unname(pr$fit),se=unname(pr$se.fit),fitted=unname(predict(b,type='link')),
 X=unname(predict(b,type='lpmatrix')),Xp=unname(Xp),sp=unname(b$sp),full_sp=unname(b$full.sp),
 covariance=unname(Xp%*%b$Vp%*%t(Xp)),freq_covariance=unname(Xp%*%b$Ve%*%t(Xp)),
 unconditional_covariance=if(is.null(b$Vc))NULL else unname(Xp%*%b$Vc%*%t(Xp)),
 edf=sum(b$edf),score=unname(b$gcv.ubre),scale=b$sig2,aic=AIC(b),loglik=as.numeric(logLik(b))),
 a[2],digits=17,auto_unbox=TRUE,null='null')
