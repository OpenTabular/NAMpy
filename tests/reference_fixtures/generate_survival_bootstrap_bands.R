# New independent R glm refits, paired effects and finite-grid max-deviation bands.
path <- "tests/reference_fixtures/survival_bootstrap_bands.csv"
if(file.exists(path)) stop("Reference exists; this script does not rebaseline it.")
indices <- as.matrix(read.csv("tests/reference_fixtures/survival_bootstrap_indices.csv"))
time <- c(.3,.8,1.4,2.1,2.6,3); entry <- c(0,.2,.1,.4,1,.5)
event <- c(1,0,1,1,0,1); weight <- c(1,2,.5,1,0,3)
offset <- c(.1,-.1,.03,-.04,.2,0); x <- c(-.8,-.2,.3,-.4,.7,1.1)
grid <- c(0,.5,1,2,4); landmark <- c(0,.5,1)
prediction_x <- c(-.6,.2,.9); prediction_offset <- c(-.03,.02,.05)
reference_x <- weighted.mean(c(-.7,.4,1),c(1,3,2))
valid <- outer(landmark,grid,"<=")
fit <- function(selected) {
    ped <- data.frame()
    for(i in selected+1) for(k in 1:2) {
        left <- c(0,1)[k]; right <- c(1,4)[k]
        exposure <- max(0,min(time[i],right)-max(entry[i],left))
        if(exposure>0) ped <- rbind(ped,data.frame(interval=factor(k,levels=1:2),x=x[i],
            event=event[i]*(time[i]>left && time[i]<=right),offset=log(exposure)+offset[i],weight=weight[i]))
    }
    unname(coef(glm(event~interval+x,offset=offset,weights=weight,family=poisson(),data=ped,
                   control=glm.control(epsilon=1e-12,maxit=100))))
}
functional <- function(beta,kind) {
    result <- matrix(NaN,3,length(grid))
    H <- function(t,r) r[1]*pmin(t,1)+r[2]*pmax(t-1,0)
    for(i in 1:3) {
        rates <- exp(c(beta[1],beta[1]+beta[2])+beta[3]*prediction_x[i]+prediction_offset[i])
        base_rates <- exp(c(beta[1],beta[1]+beta[2])+beta[3]*reference_x+prediction_offset[i])
        for(j in seq_along(grid)) if(valid[i,j]) {
            t <- grid[j]; a <- landmark[i]
            survival <- exp(-H(t,rates)+H(a,rates))
            result[i,j] <- switch(kind,
                survival=survival,
                hazard=rates[if(t<=1) 1 else 2],
                effect=survival-exp(-H(t,base_rates)+H(a,base_rates)),
                rmst=if(t==a) 0 else {
                    bounds <- c(a,if(a<1 && t>1) 1,t)
                    sum(vapply(seq_len(length(bounds)-1),function(k)
                        integrate(function(u) exp(-H(u,rates)+H(a,rates)),bounds[k],bounds[k+1],
                                  rel.tol=1e-12,abs.tol=1e-12)$value,0.))
                })
        }
    }
    result
}
betas <- rbind(fit(0:5),t(apply(indices,1,fit)))
out <- list()
for(kind in c("survival","hazard","rmst","effect")) {
    point <- functional(betas[1,],kind)
    values <- simplify2array(lapply(2:nrow(betas),function(i) functional(betas[i,],kind)))
    center <- point; transformed <- values
    if(kind=="survival") {
        positive <- valid & point<1
        center[valid] <- 0
        center[positive] <- log(-log(point[positive]))
        for(b in seq_len(dim(values)[3])) {
            transformed[,,b][valid] <- 0
            transformed[,,b][positive] <- log(-log(values[,,b][positive]))
        }
    } else if(kind=="hazard") { center <- log(point); transformed <- log(values) }
    se <- apply(transformed,c(1,2),sd)
    active <- valid & se>0
    maxima <- vapply(seq_len(dim(values)[3]),function(b)
        max(abs((transformed[,,b]-center)[active]/se[active])),0.)
    critical <- unname(quantile(maxima,.95,type=8))
    lower <- center-critical*se; upper <- center+critical*se
    if(kind=="survival") {
        lo <- exp(-exp(upper)); upper <- exp(-exp(lower)); lower <- lo
        lower[valid & point==1] <- upper[valid & point==1] <- 1
    } else if(kind=="hazard") { lower <- exp(lower); upper <- exp(upper) } else if(kind=="rmst") {
        width <- outer(landmark,grid,function(a,t) pmax(t-a,0))
        lower <- pmin(pmax(lower,0),width); upper <- pmin(pmax(upper,0),width)
    } else { lower <- pmin(pmax(lower,-1),1); upper <- pmin(pmax(upper,-1),1) }
    percentile <- apply(values,c(1,2),quantile,probs=c(.025,.975),na.rm=TRUE,type=7)
    for(b in seq_len(nrow(betas))) {
        current <- functional(betas[b,],kind)
        for(i in 1:3) for(j in seq_along(grid)) out[[length(out)+1]] <- data.frame(
            kind=kind,replicate=b-2,subject=i-1,time=grid[j],value=current[i,j],
            lower=if(b==1) percentile[1,i,j] else NaN,upper=if(b==1) percentile[2,i,j] else NaN,
            sim_lower=if(b==1) lower[i,j] else NaN,sim_upper=if(b==1) upper[i,j] else NaN,
            se=if(b==1) se[i,j] else NaN,critical=if(b==1) critical else NaN)
    }
}
write.csv(do.call(rbind,out),path,row.names=FALSE,na="NaN")
cat("Wrote independent R bootstrap-band/effect reference\n")
