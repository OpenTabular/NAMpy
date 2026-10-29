# Additional captures only: baseline-aware survival SE and source diagnostics.
if(Sys.getenv("NAMPY_REFRESH_REFERENCE_FIXTURES")!="1") stop("Set explicit reference refresh flag.")
paths <- paste0("tests/reference_fixtures/cox_",c("uncertainty.csv","diagnostics.csv"))
if(any(file.exists(paths))) stop("Reference exists; refusing to overwrite.")
library(mgcv)
data <- read.csv("tests/reference_fixtures/cox_fixed_input.csv")
data$group <- factor(data$group,levels=c("a","b"))
query <- data.frame(x=c(-.5,.2,.8),z=c(.1,-.3,.7),offset=0)
uncertainty <- diagnostics <- list()
for(stratified in c(FALSE,TRUE)) {
  current <- data
  if(stratified) current$response <- cbind(current$time,current$stratum)
  formula <- if(stratified) response~x+z else time~x+z
  fit <- gam(formula,data=current,weights=event,offset=offset,family=cox.ph(),method="REML",control=gam.control(epsilon=1e-10))
  pred <- query[rep(1:3,each=3),,drop=FALSE]
  pred$time <- rep(c(.5,1,2),3)
  if(stratified) pred$response <- cbind(pred$time,rep(c(1,2,1),each=3))
  value <- predict(fit,newdata=pred,type="response",se.fit=TRUE)
  uncertainty[[length(uncertainty)+1]] <- data.frame(stratified,survival=value$fit,se=value$se.fit)
  diagnostics[[length(diagnostics)+1]] <- data.frame(stratified,row=seq_len(nrow(data)),
    schoenfeld1=fit$family$data$Rs[,1],schoenfeld2=fit$family$data$Rs[,2],
    score1=fit$family$data$Rc[,1],score2=fit$family$data$Rc[,2],
    martingale=residuals(fit,type="martingale"),deviance=residuals(fit,type="deviance"))
}
write.csv(do.call(rbind,uncertainty),paths[1],row.names=FALSE)
write.csv(do.call(rbind,diagnostics),paths[2],row.names=FALSE)
