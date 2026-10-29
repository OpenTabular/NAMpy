# Independent mgcv fixed-smoothing Cox endpoints, including ties and strata.
prefix <- "tests/reference_fixtures/cox_fixed"
paths <- paste0(prefix,c("_input.csv",".csv"))
if(any(file.exists(paths))) stop("Reference exists; refusing to overwrite.")
library(mgcv)
i <- 0:59
source <- data.frame(x=sin(i*.7),z=cos(i*.37),group=factor(rep(c("a","b"),30)),
 time=.2+(((i*13) %% 27)/10),event=as.integer(i %% 4 != 1),stratum=rep(c(1,1,2),20),offset=.03*sin(i))

query <- data.frame(x=c(-.5,.2,.8),z=c(.1,-.3,.7),group=factor(c("a","b","a"),levels=c("a","b")),offset=0)
out <- list()
for(route in c("linear","smooth","stratified","categorical")) {
 data <- source
 if(route=="stratified") data$cox_response <- cbind(data$time,data$stratum)
 formula <- switch(route,linear=time~x+z,smooth=time~s(x,k=4)+z,
   stratified=cox_response~x+z,categorical=time~x+group)
 fit <- gam(formula,data=data,weights=event,offset=offset,family=cox.ph(),
            sp=if(route=="smooth") .4 else NULL,method="REML",control=gam.control(epsilon=1e-10))
 pred <- query[rep(1:3,each=3),,drop=FALSE]
 pred$time <- rep(c(.5,1,2),3)
 pred$stratum <- if(route=="stratified") rep(c(1,2,1),each=3) else 1
 if(route=="stratified") pred$cox_response <- cbind(pred$time,pred$stratum)
 # Predict in a batch: single-row stratified responses lose matrix dimensions
 # inside this upstream predict.gam version. The package itself is unmodified.
 survivor <- predict(fit,newdata=pred,type="response",newdata.guaranteed=TRUE)
 link <- predict(fit,newdata=pred,type="link")
 Xp <- predict(fit,newdata=pred,type="lpmatrix")
 se <- sqrt(rowSums((Xp%*%fit$Vp)*Xp))
 out[[length(out)+1]] <- data.frame(route,subject=rep(0:2,each=3),time=pred$time,
                                  survival=survivor,link=link,se=se)
}
write.csv(source,paths[1],row.names=FALSE)
write.csv(do.call(rbind,out),paths[2],row.names=FALSE)
