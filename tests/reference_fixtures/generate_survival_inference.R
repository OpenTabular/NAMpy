# New independent reference, not a refresh of existing numerical fixtures.
# R base probability laws + multivariate-normal root + R type-8 quantiles.
beta <- c(.2,.35,-.1)
V <- matrix(c(.04,.012,-.008,.012,.03,.006,-.008,.006,.02),3,3)
e <- eigen(V,symmetric=TRUE)
root <- e$vectors %*% diag(sqrt(e$values)) %*% t(e$vectors)
z <- matrix(qnorm((((1:120)*7) %% 113+.5)/113),40,3,byrow=TRUE)
draws <- sweep(z %*% root,2,beta,"+")
x <- c(-.5,.25,.8)
rx <- c(-.8,.1,.6,.95)
rw <- c(1,3,2,1)
offset <- cbind(.03*x,-.02*x)
times <- c(0,.3,1,2)
entry <- c(0,.2,.3)
valid <- outer(entry,times,"<=")

hazard <- function(eta,family) {
    result <- matrix(NA_real_,length(x),length(times))
    logsf <- function(t,i) {
        if(t==0) return(0)
        if(family=="weibull") return(-exp(exp(eta[i,2])*(log(t)-eta[i,1])))
        pnorm((log(t)-eta[i,1])/exp(eta[i,2]),lower.tail=FALSE,log.p=TRUE)
    }
    for(i in seq_along(x)) for(j in seq_along(times)) if(valid[i,j])
        result[i,j] <- -logsf(times[j],i)+logsf(entry[i],i)
    result
}
eta <- function(b,base=FALSE) cbind(b[1]+b[2]*if(base) rep(weighted.mean(rx,rw),length(x)) else x,b[3])+offset
calibrate <- function(point,samples) {
    s <- apply(samples,c(1,2),sd)
    active <- valid & s>0
    maxima <- vapply(seq_len(dim(samples)[3]),function(j)
        max(abs((samples[,,j]-point)[active]/s[active])),0.)
    list(se=s,critical=unname(quantile(maxima,.95,type=8)))
}
out <- list()
for(family in c("weibull","lognormal")) {
    h <- hazard(eta(beta),family)
    hs <- simplify2array(lapply(seq_len(nrow(draws)),function(j) hazard(eta(draws[j,]),family)))
    logh <- h; logh[valid & h==0] <- 0; logh[valid & h>0] <- log(h[valid & h>0])
    loghs <- hs
    for(j in seq_len(nrow(draws))) {
        loghs[,,j][valid & h==0] <- 0
        loghs[,,j][valid & h>0] <- log(hs[,,j][valid & h>0])
    }
    cal <- calibrate(logh,loghs)
    lower <- exp(-exp(logh+cal$critical*cal$se))
    upper <- exp(-exp(logh-cal$critical*cal$se))
    lower[valid & h==0] <- upper[valid & h==0] <- 1
    contrast <- exp(-h)-exp(-hazard(eta(beta,TRUE),family))
    cs <- simplify2array(lapply(seq_len(nrow(draws)),function(j)
        exp(-hs[,,j])-exp(-hazard(eta(draws[j,],TRUE),family))))
    ec <- calibrate(contrast,cs)
    quantiles <- apply(cs,c(1,2),quantile,probs=c(.025,.975),na.rm=TRUE,type=7)
    for(i in seq_along(x)) for(j in seq_along(times)) out[[length(out)+1]] <- data.frame(
        family=family,subject=i-1,time=times[j],estimate=exp(-h[i,j]),
        lower=lower[i,j],upper=upper[i,j],se=cal$se[i,j],critical=cal$critical,
        effect=contrast[i,j],effect_lower=quantiles[1,i,j],effect_upper=quantiles[2,i,j],
        effect_sim_lower=max(-1,contrast[i,j]-ec$critical*ec$se[i,j]),
        effect_sim_upper=min(1,contrast[i,j]+ec$critical*ec$se[i,j]),
        effect_se=ec$se[i,j],effect_critical=ec$critical)
}
path <- "tests/reference_fixtures/survival_inference.csv"
if(file.exists(path)) stop("Reference already exists; this script does not rebaseline it.")
write.csv(do.call(rbind,out),path,row.names=FALSE,na="NaN")
cat("Wrote new R reference:",path,"\n")
