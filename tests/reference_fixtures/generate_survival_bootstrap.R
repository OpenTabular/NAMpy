# Ordinary pairs resampling of complete subject records; new references only.
paths <- c("tests/reference_fixtures/survival_bootstrap_indices.csv",
           "tests/reference_fixtures/survival_bootstrap.csv")
if(any(file.exists(paths))) stop("Bootstrap references already exist; no rebaseline.")
indices <- rbind(c(0,0,1,2,3,5),c(5,5,3,2,1,0),c(2,2,2,0,1,5),
                 c(1,1,0,3,4,5),c(4,5,5,0,0,3),c(3,3,2,1,0,5))
time <- c(.3,.8,1.4,2.1,2.6,3)
entry <- c(0,.2,.1,.4,1,.5)
event <- c(1,0,1,1,0,1)
weight <- c(1,2,.5,1,0,3)
offset <- c(.1,-.1,.03,-.04,.2,0)
grid <- c(0,.5,1,2,4)
landmark <- c(0,.5)
new_offset <- c(-.05,.05)
fit <- function(selected) {
    selected <- selected+1
    ped <- data.frame()
    for(i in selected) for(k in 1:2) {
        left <- c(0,1)[k]; right <- c(1,4)[k]
        exposure <- max(0,min(time[i],right)-max(entry[i],left))
        if(exposure>0) ped <- rbind(ped,data.frame(interval=factor(k,levels=1:2),
            event=event[i]*(time[i]>left && time[i]<=right),
            offset=log(exposure)+offset[i],weight=weight[i]))
    }
    model <- glm(event~0+interval,offset=offset,weights=weight,family=poisson(),data=ped)
    unname(exp(coef(model)))
}
curves <- function(rate) {
    result <- matrix(NaN,2,length(grid))
    H <- function(t,r) r[1]*pmin(t,1)+r[2]*pmax(t-1,0)
    for(i in 1:2) for(j in seq_along(grid)) if(grid[j]>=landmark[i])
        result[i,j] <- exp(-H(grid[j],rate*exp(new_offset[i]))+H(landmark[i],rate*exp(new_offset[i])))
    result
}
rates <- rbind(fit(0:5),t(apply(indices,1,fit)))
samples <- simplify2array(lapply(2:nrow(rates),function(i) curves(rates[i,])))
limits <- apply(samples,c(1,2),quantile,probs=c(.025,.975),na.rm=TRUE,type=7)
out <- list()
for(b in seq_len(nrow(rates))) for(i in 1:2) for(j in seq_along(grid))
    out[[length(out)+1]] <- data.frame(replicate=b-2,subject=i-1,time=grid[j],
        rate1=rates[b,1],rate2=rates[b,2],survival=curves(rates[b,])[i,j],
        lower=if(b==1) limits[1,i,j] else NaN,upper=if(b==1) limits[2,i,j] else NaN)
write.csv(indices,paths[1],row.names=FALSE)
write.csv(do.call(rbind,out),paths[2],row.names=FALSE,na="NaN")
cat("Wrote independent R glm subject-bootstrap references\n")
