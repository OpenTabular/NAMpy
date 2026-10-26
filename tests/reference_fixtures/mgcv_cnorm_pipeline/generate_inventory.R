# Run from repository root. Actual source acceptance/errors, not a fallback.
library(mgcv);library(jsonlite)
root <- 'tests/reference_fixtures/mgcv_cnorm_pipeline'
files <- c('candidate_dense_example.json','candidate_dense_workflow.json','inputs.json')
results <- lapply(files,function(file) {
 d <- as.data.frame(fromJSON(file.path(root,file))); d$bound <- as.numeric(d$bound)
 warnings <- character()
 result <- tryCatch(withCallingHandlers({
  b <- gam(cbind(y,bound) ~ z+s(x,bs='cr',k=6,sp=.8)+offset(off),data=d,family=cnorm(theta=1),method='REML')
  nd <- data.frame(x=seq(-1.1,1.1,length.out=19));nd$z <- .4+cos(1.3*nd$x);nd$off<-.15*sin(.7*nd$x)
  pr <- predict(b,nd,se.fit=TRUE)
  list(accepted=TRUE,coef=coef(b),Vp=b$Vp,edf=b$edf,sp=b$sp,full_sp=b$full.sp,scale=b$sig2,
       deviance=deviance(b),iter=b$iter,converged=b$converged,mu=fitted(b),prediction=pr$fit,se=pr$se.fit,
       y=as.numeric(b$y),bounds=attr(b$y,"censor"),theta=b$family$getTheta(),Dd=b$family$Dd(b$y,fitted(b),b$family$getTheta(),b$prior.weights,2))
 },warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e)list(accepted=FALSE,error=conditionMessage(e)))
 result$input <- file;result$warnings <- warnings;result
})
write_json(list(R=as.character(getRversion()),mgcv=as.character(packageVersion('mgcv')),cases=results),file.path(root,'candidate_inventory.json'),digits=17,auto_unbox=TRUE,na='string',pretty=TRUE)
