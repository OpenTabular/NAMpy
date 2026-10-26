suppressPackageStartupMessages(library(mgcv));library(jsonlite)
stopifnot(as.character(packageVersion('mgcv'))=='1.9.4')
a<-commandArgs(TRUE);p<-fromJSON(a[1]);d<-as.data.frame(p$data);nd<-as.data.frame(p$newdata);dd<-as.data.frame(p$derivative_data)
ans<-list();C<-NULL
smooth.construct.custom.smooth.spec<-function(object,data,knots){
 class(object)<-'bs.smooth.spec';object$bs<-'bs';z<-smooth.construct(object,data,knots)
 if(!is.null(C))z$C<-C
 z
}
for(mname in c('single','multiple'))for(cname in c('ordinary','one','two','empty'))for(select in c(FALSE,TRUE)){
 m<-if(mname=='single')c(3,2) else c(3,2,1)
 C<-switch(cname,ordinary=NULL,one=matrix(1,1,8),two=rbind(rep(1,8),0:7),empty=matrix(0,0,8))
 spec<-s(x,bs='custom',k=8,m=m)
 raw<-smoothCon(spec,d,absorb.cons=FALSE,null.space.penalty=select)[[1]]
 con<-smoothCon(spec,d,absorb.cons=TRUE,null.space.penalty=select)[[1]]
 deriv<-lapply(1:3,function(q){z<-con;z$deriv<-q;unname(PredictMat(z,dd))})
 result<-list(raw_X=unname(raw$X),raw_S=lapply(raw$S,unname),C=unname(raw$C),X=unname(con$X),S=lapply(con$S,unname),rank=con$rank,null_dim=con$null.space.dim,P=unname(PredictMat(con,nd)),derivatives=deriv,fits=list())
 for(method in c('fixed','REML','GCV',if(mname=='multiple'&&cname=='empty'&&select)'GCV_matched_controls')){
  form<-as.formula(sprintf("y~s(x,bs='custom',k=8,m=c(%s))",paste(m,collapse=',')))
  b<-gam(form,data=d,sp=if(method=='fixed')rep(.7,length(con$S))else NULL,method=if(startsWith(method,'GCV'))'GCV.Cp' else 'REML',select=select,control=gam.control(mgcv.tol=if(method=='GCV_matched_controls')1e-12 else 1e-7))
  pr<-predict(b,nd,se.fit=TRUE);P<-predict(b,nd,type='lpmatrix')
  result$fits[[method]]<-list(sp=if(method=='fixed')rep(.7,length(con$S))else unname(b$sp),fit=unname(pr$fit),se=unname(pr$se.fit),train_fit=unname(fitted(b)),edf=sum(b$edf),score=unname(b$gcv.ubre),X=unname(model.matrix(b)),P=unname(P),covariance=unname(P%*%b$Vp%*%t(P)))
 }
 ans[[paste(mname,cname,select,sep='/')]]<-result
}
write_json(list(version=as.character(packageVersion('mgcv')),R=as.character(getRversion()),cases=ans),a[2],digits=17,auto_unbox=TRUE)
