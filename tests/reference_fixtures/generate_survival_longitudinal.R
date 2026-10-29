# Independent Poisson survival likelihood with changing covariates and exposure.
input <- "tests/reference_fixtures/survival_longitudinal_input.csv"
output <- "tests/reference_fixtures/survival_longitudinal.csv"
if(file.exists(input)||file.exists(output)) stop("Reference exists; refusing to overwrite.")
subject <- rep(letters[1:8],each=2)
change <- c(.3,.5,.4,.8,.6,.7,1.1,.9)
exit <- c(.9,1.3,1.5,1.8,2.2,2.5,2.8,2.9)
event <- c(1,0,1,1,0,1,1,1)
x1 <- c(-.8,.3,-.4,.9,-.2,.6,-.6,.4)
x2 <- c(.4,-.3,.8,-.2,.7,-.5,.3,-.7)
weight <- c(1,2,.5,1,2,1,.5,1)
source <- data.frame(subject,start=as.vector(rbind(rep(0,8),change)),
 stop=as.vector(rbind(change,exit)),event=as.vector(rbind(rep(0,8),event)),
 x=as.vector(rbind(x1,x2)),weight=rep(weight,each=2),offset=seq(-.04,.05,length.out=16))
cut <- c(0,1,3)
ped <- data.frame()
for(i in seq_len(nrow(source))) for(j in 1:2) {
  exposure <- max(0,min(source$stop[i],cut[j+1])-max(source$start[i],cut[j]))
  if(exposure>0) ped <- rbind(ped,data.frame(interval=factor(j,levels=1:2),
   event=source$event[i]*(source$stop[i]>cut[j]&&source$stop[i]<=cut[j+1]),
   x=source$x[i],weight=source$weight[i],offset=log(exposure)+source$offset[i],exposure))
}
fit <- glm(event~interval+x,data=ped,family=poisson(),weights=weight,offset=offset,
           control=glm.control(epsilon=1e-12,maxit=100))
stopifnot(fit$converged)
# Full hypothetical paths from zero to 3, with change at .7 / 1.4.
grid <- c(0,.7,1,1.4,3)
pathx <- rbind(c(-.5,.8,.8,.8),c(-.2,-.2,-.2,.4))
interval <- c(1,1,2,2)
new <- data.frame(interval=factor(rep(interval,2),levels=1:2),x=as.vector(t(pathx)))
new$offset <- 0
eta <- matrix(predict(fit,newdata=new,type="link"),2,4,byrow=TRUE)
rates <- exp(eta)
continuous_ll <- sum(ped$weight*(ped$event*(predict(fit,type="link")-log(ped$exposure))-exp(predict(fit,type="link"))))
out <- list()
for(i in 1:2) for(t in c(.5,1,2,3)) out[[length(out)+1]] <- data.frame(subject=i-1,time=t,
 survival=exp(-sum(rates[i,]*pmax(0,pmin(t,grid[-1])-grid[-length(grid)]))),
 rate1=rates[i,1],rate2=rates[i,2],rate3=rates[i,3],rate4=rates[i,4],
 loglik=continuous_ll,score=continuous_ll/sum(weight))
write.csv(source,input,row.names=FALSE)
write.csv(do.call(rbind,out),output,row.names=FALSE)
