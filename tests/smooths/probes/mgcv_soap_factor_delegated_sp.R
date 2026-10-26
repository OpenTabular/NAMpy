suppressPackageStartupMessages(library(mgcv));library(jsonlite)
stopifnot(as.character(packageVersion('mgcv'))=='1.9.4')
d<-expand.grid(x=c(-.8,-.4,0,.4,.8),y=c(-.8,-.4,0,.4,.8))
d$f<-factor(rep('a',nrow(d)));d$response<-.2+.3*d$x-.1*d$y
nd<-data.frame(x=c(-.6,.2,.6),y=c(.2,-.6,.6),f=factor(rep('a',3)))
bnd<-list(list(x=c(-1,1,1,-1,-1),y=c(-1,-1,1,1,-1)))
kn<-list(x=c(-.5,.5,-.5,.5),y=c(-.5,-.5,.5,.5));cases<-list()
for(mode in c('fs','sz'))for(bs in c('so','sw')) {
 sp<-if(bs=='so')c(.2,.4) else .2
 expr<-sprintf("s(x,y,bs='%s',k=8,xt=list(bs='%s',bnd=bnd,nmax=30),sp=c(%s))",mode,bs,paste(sp,collapse=','))
 b<-gam(as.formula(paste('response~',expr)),data=d,knots=kn,method='REML')
 pr<-predict(b,nd,se.fit=TRUE);P<-predict(b,nd,type='lpmatrix')
 cases[[paste(mode,bs,sep='/')]]<-list(term_sp=sp,free_sp=b$sp,full_sp=b$full.sp,
  fitted=unname(fitted(b)),prediction=unname(pr$fit),se=unname(pr$se.fit),
  covariance=unname(P%*%b$Vp%*%t(P)),edf=sum(b$edf),
  X=unname(predict(b,type='lpmatrix')),Xp=unname(P),S=lapply(b$smooth[[1]]$S,unname))
}
write_json(list(version=as.character(packageVersion('mgcv')),data=d,newdata=nd,knots=kn,
 levels=levels(d$f),boundary=bnd,cases=cases),commandArgs(TRUE)[1],digits=17,auto_unbox=TRUE,null='null')
