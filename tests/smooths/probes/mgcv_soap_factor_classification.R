suppressPackageStartupMessages(library(mgcv));library(jsonlite)
stopifnot(as.character(packageVersion('mgcv'))=='1.9.4')
# Deliberate bounded source classification, no source patches or coordinate fixes.
d<-expand.grid(x=c(-.8,-.4,0,.4,.8),y=c(-.8,-.4,0,.4,.8),f=c('a','b','c'))
d<-d[!(abs(d$x)<.2&abs(d$y)<.2),];d$f<-factor(d$f,levels=c('a','b','c'))
d$response<-.2+.3*d$x-.1*d$y+as.integer(d$f)*.07*d$x
nd<-data.frame(x=c(-.6,.2,.6),y=c(.2,-.6,.6),f=factor(c('c','a','b'),levels=levels(d$f)))
kn<-list(x=c(-.5,.5,-.5,.5),y=c(-.5,-.5,.5,.5))
outer<-list(x=c(-1,1,1,-1,-1),y=c(-1,-1,1,1,-1))
hole<-list(x=c(-.15,-.15,.15,.15,-.15),y=c(-.15,.15,.15,-.15,-.15))
domains<-list(free=list(outer),mixed=list(outer,c(hole,list(f=rep(.4,5)))),
 multiple_free=list(outer,hole),known=list(c(outer,list(f=rep(.2,5)))),
 all_known=list(c(outer,list(f=rep(.2,5))),c(hole,list(f=rep(.4,5)))),
 outer_known=list(c(outer,list(f=rep(.2,5))),hole))
probe<-function(expr) {
 warns<-character()
 result<-tryCatch(withCallingHandlers(expr,warning=function(w){warns<<-c(warns,conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e)e)
 if(inherits(result,'error'))list(ok=FALSE,error=conditionMessage(result),call=paste(deparse(conditionCall(result)),collapse=' '),warnings=warns) else list(ok=TRUE,value=result,warnings=warns)
}
meta<-function(sm)list(ncol=ncol(sm$X),npen=length(sm$S),rank=unname(sm$rank),null_space_dim=sm$null.space.dim,
 penalty_rank=vapply(sm$S,function(S)qr(S)$rank,1L),offset=attr(sm$X,"offset"))
cases<-list()
for(domain in names(domains))for(bs in c('so','sf','sw')) {
 bnd<-domains[[domain]]
 expr<-sprintf("s(x,y,bs='%s',k=8,xt=list(bnd=bnd,nmax=30))",bs)
 base<-probe(meta(smoothCon(eval(parse(text=expr)),d,knots=kn,absorb.cons=FALSE)[[1]]))
 for(mode in c('fs','sz'))for(factor_present in c(TRUE,FALSE)) {
  features<-if(factor_present)'x,y,f' else 'x,y'
  expr<-sprintf("s(%s,bs='%s',k=8,xt=list(bs='%s',bnd=bnd,nmax=30))",features,mode,bs)
  constructor<-probe(meta(smooth.construct(eval(parse(text=expr)),d,knots=kn)))
  constrained<-probe(meta(smoothCon(eval(parse(text=expr)),d,knots=kn,absorb.cons=TRUE)[[1]]))
  fit<-probe({
   form<-as.formula(paste('response~',expr))
   sm<-smoothCon(eval(parse(text=expr)),d,knots=kn,absorb.cons=TRUE)[[1]]
   b<-gam(form,data=d,knots=kn,method='REML',sp=rep(.3,length(sm$S)))
   pr<-predict(b,nd,se.fit=TRUE);P<-predict(b,nd,type='lpmatrix')
   list(fitted=unname(fitted(b)),prediction=unname(pr$fit),se=unname(pr$se.fit),
    covariance=unname(P%*%b$Vp%*%t(P)),edf=sum(b$edf),sp=b$sp,full_sp=b$full.sp,
    X=unname(predict(b,type='lpmatrix')),Xp=unname(P),S=lapply(b$smooth[[1]]$S,unname))
  })
  key<-paste(mode,bs,domain,sep='/');if(!factor_present)key<-paste('plain',key,sep='/')
  cases[[key]]<-list(mode=mode,basis=bs,domain=domain,boundary=bnd,
   base=base,constructor=constructor,constrained=constrained,fit=fit)
 }
}
write_json(list(version=as.character(packageVersion('mgcv')),data=d,newdata=nd,knots=kn,
 levels=levels(d$f),cases=cases),commandArgs(TRUE)[1],digits=17,auto_unbox=TRUE,null='null')
