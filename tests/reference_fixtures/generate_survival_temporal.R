# New mgcv references for right-endpoint time/covariate smooth interactions.
if(Sys.getenv("NAMPY_REFRESH_REFERENCE_FIXTURES")!="1") stop("Set explicit reference refresh flag.")
prefix <- "tests/reference_fixtures/survival_temporal"
paths <- paste0(prefix,c("_input.csv",".csv"))
if(any(file.exists(paths))) stop("Reference exists; refusing to overwrite.")
library(mgcv)
i <- 0:89
source <- data.frame(x=sin(i*.7),time=.2+((i*13) %% 47)/10,
                     event=as.integer(i %% 4 != 1),entry=rep(c(0,.05,.1),30),
                     weight=rep(c(1,2,.5),30),offset=.04*sin(i*.7))
cuts <- 0:5
ped <- data.frame()
for(j in 1:nrow(source)) for(k in 1:5) {
  exposure <- max(0,min(source$time[j],cuts[k+1])-max(source$entry[j],cuts[k]))
  if(exposure>0) ped <- rbind(ped,data.frame(x=source$x[j],survival_time=cuts[k+1],
    event=source$event[j]*(source$time[j]>cuts[k]&&source$time[j]<=cuts[k+1]),
    offset=log(exposure)+source$offset[j],weight=source$weight[j]))
}
query <- expand.grid(survival_time=1:5,x=c(-.5,0,.8))
results <- list()
for(route in c("fixed","reml")) {
  fit <- gam(event~s(survival_time,k=4)+x+ti(survival_time,x,k=c(4,4)),
    data=ped,family=poisson(),weights=weight,offset=offset,
    sp=if(route=="fixed") c(.4,.3,.2) else NULL,method="REML",control=gam.control(epsilon=1e-10))
  link <- as.numeric(predict(fit,newdata=query,type="link"))
  results[[length(results)+1]] <- data.frame(route,link,sp1=fit$sp[1],sp2=fit$sp[2],sp3=fit$sp[3],loglik=as.numeric(logLik(fit)))
}
write.csv(source,paths[1],row.names=FALSE)
write.csv(do.call(rbind,results),paths[2],row.names=FALSE)
