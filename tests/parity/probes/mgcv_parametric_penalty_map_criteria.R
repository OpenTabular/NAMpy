suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1],simplifyVector=FALSE)
d<-as.data.frame(lapply(p$data,unlist));d$y<-d[[paste0('y_',p$family)]];d$B<-I(as.matrix(d[,c('x','u','v')]))
P<-lapply(p$penalties,function(S)do.call(rbind,lapply(S,unlist)))
P$L<-do.call(rbind,lapply(p$L,unlist));P$sp<-unlist(p$sp)
fam<-if(p$family=='gaussian')gaussian() else poisson()
fam<-mgcv:::fix.family.link(fam);fam<-mgcv:::fix.family.var(fam);fam$qapprox<-FALSE
G<-gam(y~B,data=d,family=fam,paraPen=list(B=P),method='REML',fit=FALSE)
space<-mgcv:::totalPenaltySpace(G$S,G$H,G$off,ncol(G$X))
roots<-mgcv:::mini.roots(G$S,G$off,ncol(G$X),G$rank)
UrS<-lapply(roots,function(root)t(space$Y)%*%root);U1<-cbind(space$Y,space$Z)
fixed<-unlist(p$sp)>=0;ev<-unlist(p$eval_sp)[!fixed]
L<-G$L;if(is.null(L))L<-diag(length(G$S))
nei<-list(a=seq_len(nrow(d)),ma=seq_len(nrow(d)),d=seq_len(nrow(d)),md=seq_len(nrow(d)),jackknife=-1)
fit<-mgcv:::gam.fit3(x=G$X,y=G$y,sp=drop(L%*%log(ev)+G$lsp0),
 Eb=space$E,UrS=UrS,weights=G$w,U1=U1,Mp=ncol(space$Z),family=fam,
 control=gam.control(epsilon=1e-10),offset=rep(0,nrow(d)),null.coef=rep(0,ncol(G$X)),
 scale=if(p$family=='gaussian')-1 else 1,deriv=if(p$criterion=='NCV')1 else 2,scoreType=p$criterion,nei=nei)
score<-fit[[p$criterion]];gradient<-fit[[paste0(p$criterion,'1')]];hessian<-fit[[paste0(p$criterion,'2')]]
write_json(list(score=unname(score),gradient=drop(t(L)%*%gradient),
 hessian=if(is.null(hessian))NULL else unname(t(L)%*%hessian%*%L)),a[2],digits=17,auto_unbox=TRUE,null='null')
