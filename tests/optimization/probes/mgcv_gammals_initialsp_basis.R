if (Sys.getenv("NAMPY_REFRESH_REFERENCE_FIXTURES") != "1") stop("Set NAMPY_REFRESH_REFERENCE_FIXTURES=1 to capture references")
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1]);x<-as.matrix(p$X);y<-p$y;E<-as.matrix(p$E)
S<-list(as.matrix(p$S0),as.matrix(p$S1));rank<-p$rank;off<-p$off
family<-mgcv:::fix.family(gammals());weights<-rep(1,length(y))
attr(x,"lpi")<-list(1:6,7L);offset<-list(rep(0,length(y)),rep(0,length(y)))
initial<-function(x)mgcv:::initial.spg(x,y,weights,family,S,rank,off,offset=offset,E=E)
original<-initial(x);x[,5]<- -x[,5];flipped<-initial(x)
write_json(list(original=original,flipped=flipped,version=as.character(packageVersion('mgcv')),R=as.character(getRversion())),a[2],digits=17,auto_unbox=TRUE)
