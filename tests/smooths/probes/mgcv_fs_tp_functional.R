# Capture source fs/tp operators and their actual nat.param inputs.
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1]);d<-p$data;nd<-p$newdata
d$f<-factor(d$f,levels=p$levels);nd$f<-factor(nd$f,levels=p$levels)
expr<-sprintf("s(X,f,bs='fs',xt=list(bs='tp'),k=6,m=%d%s)",p$m,if(p$kind=='unit')''else',by=L')
spec<-eval(parse(text=expr))
# Inspect the unmodified source inputs to type=1 natural parameterization.
trace('nat.param',tracer=quote({assign('.audit_X',X,.GlobalEnv);assign('.audit_S',S,.GlobalEnv);assign('.audit_rank',rank,.GlobalEnv)}),where=asNamespace('mgcv'),print=FALSE)
raw<-mgcv:::smooth.construct3(spec,d,NULL)
untrace('nat.param',where=asNamespace('mgcv'))
nat<-mgcv:::nat.param(.audit_X,.audit_S,rank=.audit_rank,type=1)
sm<-smoothCon(spec,d,n=length(d$response),absorb.cons=TRUE)[[1]]
form<-as.formula(paste('response~',expr))
b<-gam(form,data=d,method='REML',sp=if(p$method=='fixed')rep(.3,length(sm$S))else NULL,
 control=gam.control(epsilon=1e-10,newton=list(conv.tol=1e-10)))
pr<-predict(b,nd,se.fit=TRUE);P<-predict(b,nd,type='lpmatrix')
write_json(list(version=as.character(packageVersion('mgcv')),data=p$data,newdata=p$newdata,levels=p$levels,
 m=p$m,null_dim=raw$base$null.space.dim,sp=unname(if(p$method=='fixed')rep(.3,length(sm$S))else b$sp),
 X=unname(predict(b,type='lpmatrix')),Xp=unname(P),S=unname(lapply(b$smooth[[1]]$S,unname)),
 S_scale=sm$S.scale,raw_S=unname(lapply(raw$S,unname)),raw_X=unname(raw$X),
 nat_X=.audit_X,nat_S=.audit_S,nat_rank=.audit_rank,nat=nat,
 fit=unname(pr$fit),se=unname(pr$se.fit),fitted=unname(fitted(b)),edf=sum(b$edf),
 covariance=unname(P%*%b$Vp%*%t(P)),score=unname(b$gcv.ubre),converged=b$converged),a[2],digits=17,auto_unbox=TRUE)
