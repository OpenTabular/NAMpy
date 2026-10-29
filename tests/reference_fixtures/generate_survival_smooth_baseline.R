# Independent mgcv Poisson reference; refuses to overwrite any existing fixture.
path <- "tests/reference_fixtures/survival_smooth_baseline.csv"
if (file.exists(path)) stop("Reference exists; refusing to rebaseline.")
library(mgcv)
n <- 60
x <- sin((0:(n-1))*.7)
time <- .2 + (((0:(n-1))*13) %% 47)/10
entry <- rep(c(0,.05,.1),length.out=n)
event <- as.integer((0:(n-1)) %% 4 != 1)
weight <- rep(c(1,2,.5,0),length.out=n)
offset <- .04*x
cuts <- 0:5
stopifnot(max(time)<max(cuts), sum(event*weight)>10)
ped <- data.frame()
for(i in seq_len(n)) for(k in 1:5) {
  exposure <- max(0,min(time[i],cuts[k+1])-max(entry[i],cuts[k]))
  if(exposure>0) ped <- rbind(ped,data.frame(survival_time=cuts[k+1],x=x[i],
    event=event[i]*(time[i]>cuts[k] && time[i]<=cuts[k+1]),
    offset=log(exposure)+offset[i],weight=weight[i]))
}
pred <- expand.grid(interval=1:5,subject=0:2)
pred$survival_time <- pred$interval
pred$x <- c(-.5,0,.8)[pred$subject+1]
values <- list()
for(route in c("fixed","reml")) {
  # All five distinct endpoints: no roundoff-dependent truncated eigenbasis.
  fit <- if(route=="fixed") gam(event~s(survival_time,k=5)+x,data=ped,
    weights=weight,offset=offset,family=poisson(),sp=.4,method="REML",control=gam.control(epsilon=1e-12,newton=list(conv.tol=1e-10))) else
    gam(event~s(survival_time,k=5)+x,data=ped,weights=weight,offset=offset,
    family=poisson(),method="REML",control=gam.control(epsilon=1e-12,newton=list(conv.tol=1e-10)))
  eta <- predict(fit,newdata=pred,type="link") # known user offsets explicitly zero
  rate <- matrix(exp(eta),3,5,byrow=TRUE)
  for(i in 1:3) for(j in 1:5) values[[length(values)+1]] <- data.frame(
    route=route,subject=i-1,interval=j-1,link=eta[(i-1)*5+j],rate=rate[i,j],
    survival=exp(-sum(rate[i,1:j])),sp=fit$sp[1],loglik=as.numeric(logLik(fit)))
}
write.csv(do.call(rbind,values),path,row.names=FALSE)
