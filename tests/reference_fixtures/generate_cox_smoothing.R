# New independent mgcv Cox automatic-smoothing endpoints. Never overwrite.
if(Sys.getenv("NAMPY_REFRESH_REFERENCE_FIXTURES")!="1") stop("Set explicit reference refresh flag.")
prefix <- "tests/reference_fixtures/cox_smoothing"
paths <- paste0(prefix,c("_input.csv",".csv"))
if(any(file.exists(paths))) stop("Reference exists; refusing to overwrite.")
library(mgcv)
i <- 0:119
x <- sin(i*.71); z <- cos(i*.33)
u <- ((i*37) %% 119+.5)/120
e <- -log(u)/exp(1.5*sin(3*x)+.2*z)
censor <- .5+((i*53) %% 113)/60
data <- data.frame(x=x,z=z,time=round(pmin(e,censor),3)+.001,
                   event=as.integer(e<=censor),stratum=rep(1:2,60),offset=.03*sin(i))
query <- data.frame(x=c(-.7,.1,.8),z=c(.1,-.2,.4),offset=0)
results <- list()
for(stratified in c(FALSE,TRUE)) {
 current <- data
 if(stratified) current$response <- cbind(current$time,current$stratum)
 formula <- if(stratified) response~s(x,k=5)+z else time~s(x,k=5)+z
 fit <- gam(formula,data=current,weights=event,offset=offset,family=cox.ph(),method="REML",
            control=gam.control(epsilon=1e-10))
 pred <- query[rep(1:3,each=3),,drop=FALSE]
 pred$time <- rep(c(.1,.3,.5),3)
 if(stratified) pred$response <- cbind(pred$time,rep(c(1,2,1),each=3))
 X <- predict(fit,newdata=pred,type="lpmatrix")
 results[[length(results)+1]] <- data.frame(stratified=stratified,sp=fit$sp[1],
   score=as.numeric(fit$gcv.ubre),link=as.numeric(predict(fit,newdata=pred,type="link")),
   se=sqrt(rowSums((X%*%fit$Vp)*X)),survival=as.numeric(predict(fit,newdata=pred,type="response")))
}
write.csv(data,paths[1],row.names=FALSE)
write.csv(do.call(rbind,results),paths[2],row.names=FALSE)
