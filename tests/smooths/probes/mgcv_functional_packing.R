suppressPackageStartupMessages(library(mgcv));library(jsonlite)
set.seed(984);n<-85;q<-4;nt<-13
ids<-matrix(sample(1:36,n*q,TRUE,prob=seq(36,1)^2),n,q);ids[1:36]<-1:36
ni<-matrix(sample(1:36,nt*q,TRUE),nt,q)
grid<-seq(-1,1,length.out=36);zg<-runif(36,-1,1)
X<-matrix(grid[ids],n,q);Z<-matrix(zg[ids],n,q);NX<-matrix(grid[ni],nt,q);NZ<-matrix(zg[ni],nt,q)
L<-matrix(runif(n*q,.1,1),n,q);NL<-matrix(runif(nt*q,.1,1),nt,q);NL[3,]<-0
w<-runif(n,.7,1.3);w[c(4,18)]<-0;noise<-rnorm(n,sd=.12);cases<-list()
for(bs in c('tp','ts','ds','gp','sos','cr','cs','cc','ps','cp','bs','ad','ad2')) for(kind in c('weighted','unweighted','constant')) for(method in c('fixed','REML')) {
 if(method=='REML' && kind!='weighted' && bs!='ad2') next
 if(bs=='ad2' && (kind!='unweighted')) next
 dim<-if(bs %in% c('tp','ds','gp','sos','ad2'))2 else 1
 WL<-if(kind=='constant')L/rowSums(L) else L
 WNL<-if(kind=='constant'){tmp<-NL;tmp[3,]<-1;tmp/rowSums(tmp)}else NL
 W<-if(kind=='unweighted')matrix(1,n,q) else WL
 y<-.3+rowSums(W*(sin(X)+if(dim==2).3*Z^2 else 0))+noise
 data<-list(y=y,X=X,Z=Z,L=WL,w=w);nd<-list(X=NX,Z=NZ,L=WNL)
 bsn<-if(bs=='ad2')'ad' else bs
 extra<-if(bs=='ad2')',k=c(5,5),m=3' else if(bs=='ad')',k=10,m=3' else ',k=10'
 form<-as.formula(sprintf("y~s(%s,bs='%s'%s%s)",if(dim==2)'X,Z' else 'X',bsn,extra,if(kind=='unweighted')'' else ',by=L'))
 ns<-if(bs=='ad2')9 else if(bs=='ad')3 else 1
 b<-if(method=='fixed')gam(form,data=data,weights=w,sp=rep(.4,ns),method='REML')else gam(form,data=data,weights=w,method='REML')
 P<-predict(b,nd,type='lpmatrix');pr<-predict(b,nd,se.fit=TRUE)
 cases[[paste(bs,kind,method,sep='/')]]<-list(data=data,newdata=nd,coef=unname(coef(b)),X=unname(predict(b,type='lpmatrix')),Xp=P,S=lapply(b$smooth[[1]]$S,unname),fit=unname(pr$fit),se=unname(pr$se.fit),fitted=unname(fitted(b)),kernel=unname(P%*%b$Vp%*%t(P)),edf=sum(b$edf),sp=unname(if(method=='fixed')rep(.4,ns)else b$sp))
}
# Explicit setup identity captures column-major order and unpacking.
a<-array(c(1,2,1,3,1,2,4,5,4,6,4,5),c(3,2,2))
pack<-mgcv:::ExtractData(s(A,B,bs='tp'),list(A=a[,,1],B=a[,,2]),NULL)$data
write_json(list(version=as.character(packageVersion('mgcv')),cases=cases,helper=list(A=a[,,1],B=a[,,2],points=cbind(pack$A,pack$B),index=attr(pack,'index'))),commandArgs(TRUE)[1],digits=17,auto_unbox=TRUE)
