suppressPackageStartupMessages(library(mgcv));library(jsonlite)
stopifnot(as.character(packageVersion('mgcv'))=='1.9.4')
set.seed(621);n<-105;q<-3;nt<-17
ans<-list()
for(matrix_arg in TRUE)for(kind in c('factor','mixed'))for(fun in c('te','ti','t2'))for(method in c('fixed','REML'))for(precision in FALSE) if(!precision || (kind=='factor' && !matrix_arg)) for(repeated in c(FALSE,TRUE)) if(!repeated || (matrix_arg && kind=='mixed' && !precision)) {
 make<-function(n) {
  x<-runif(n*q,.2,1.5);z<-runif(n*q,-1,1);f<-sample(c("01","1","2","3"),n*q,TRUE);L<-runif(n*q,.1,1)
  if(matrix_arg) {x<-matrix(x,n,q);z<-matrix(z,n,q);f<-factor(f,levels=c("01","1","2","3"));dim(f)<-c(n,q);L<-matrix(L,n,q)} else {x<-x[1:n];z<-z[1:n];f<-factor(f[1:n],levels=c("01","1","2","3"));L<-L[1:n]}
  list(x=x,z=z,f=f,L=L,o=seq(-.1,.2,length.out=n))
 }
 d<-make(n);nd<-make(nt)
 if(repeated) for(name in c('x','z','f')) d[[name]][,2:3]<-rep(d[[name]][1:35,1],length.out=n*2)
 effect<-if(kind=='numeric') d$x else (match(d$f,c("01","1","2","3"))-.7)*if(kind=='mixed')d$x else 1
 if(matrix_arg)effect<-matrix(effect,n,q)
 signal<-effect*(.4+sin(d$z*2))
 d$response<-.3+d$o+if(matrix_arg)rowSums(d$L*signal)+rnorm(n,sd=.1) else signal+rnorm(n,sd=.1)
 variables<-if(kind=='numeric')'x,z' else if(kind=='factor')'f,z' else 'x,f,z'
 expr<-sprintf("%s(%s,bs=c('re','cr'),d=c(%d,1),k=c(4,5)%s)",fun,variables,if(kind=='mixed')2 else 1,if(matrix_arg)',by=L' else '')
 if(precision)expr<-sub("\\)$",",xt=list(list(S=diag(1:4),rank=4),NULL))",expr)
 result<-tryCatch({
  sm<-smoothCon(eval(parse(text=expr)),d,n=n,absorb.cons=TRUE)[[1]]
  form<-as.formula(paste('response~',expr,'+offset(o)'))
  sp<-seq(.4,.8,length.out=length(sm$S))
  b<-if(method=='fixed')gam(form,data=d,sp=sp,method='REML') else gam(form,data=d,method='REML')
  P<-predict(b,nd,type='lpmatrix');pr<-predict(b,nd,se.fit=TRUE)
  unknown<-NULL
  if(kind!='numeric') {
   ud<-nd;values<-as.character(ud$f)
   if(matrix_arg) values[seq(1,nt*q,by=nt)]<-'new' else values[1]<-'new'
   ud$f<-factor(values,levels=c(c("01","1","2","3"),'new'));if(matrix_arg)dim(ud$f)<-c(nt,q)
   up<-suppressWarnings(predict(b,ud,se.fit=TRUE))
   unknown<-list(fit=unname(up$fit),se=unname(up$se.fit))
  }
  serial_d<-d;serial_nd<-nd
  if(matrix_arg){serial_d$f<-matrix(as.character(d$f),n,q);serial_nd$f<-matrix(as.character(nd$f),nt,q)}
  list(data=serial_d,newdata=serial_nd,sp=if(method=='fixed')sp else unname(b$sp),fit=unname(pr$fit),se=unname(pr$se.fit),
   fitted=unname(fitted(b)),edf=sum(b$edf),covariance=unname(P%*%b$Vp%*%t(P)),raw_X=unname(sm$X),raw_S=unname(lapply(sm$S,unname)),X=unname(predict(b,type='lpmatrix')),Xp=unname(P),rank=sm$rank,null_dim=sm$null.space.dim,
   formula=deparse(form),unknown=unknown,precision=precision)
 },error=function(e) list(error=conditionMessage(e)))
 key<-paste(matrix_arg,kind,fun,method,sep='/');if(precision)key<-paste0(key,'/precision');if(repeated)key<-paste0(key,'/repeated')
 ans[[key]]<-result
}
write_json(list(version=as.character(packageVersion('mgcv')),cases=ans),commandArgs(TRUE)[1],digits=17,auto_unbox=TRUE)
