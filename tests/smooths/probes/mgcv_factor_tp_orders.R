# Scalar FS/SZ nondefault thin-plate derivative orders.
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1]);d<-p$data
d$X<-as.numeric(d$X[,1]);d$f<-factor(d$f,levels=p$levels)
expr<-sprintf("s(X,f,bs='%s',xt=list(bs='%s'),k=6,m=%d)",p$mode,p$base,p$m)
sm<-smoothCon(eval(parse(text=expr)),d,n=length(d$response),absorb.cons=TRUE)[[1]]
write_json(list(version=as.character(packageVersion('mgcv')),data=d,levels=p$levels,
 X=unname(sm$X),S=unname(lapply(sm$S,unname)),rank=sm$rank,null_dim=sm$null.space.dim,
 base_rank=sm$base$rank,base_null_dim=sm$base$null.space.dim),a[2],digits=17,auto_unbox=TRUE)
