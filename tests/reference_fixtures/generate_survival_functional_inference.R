# Independent hazard/RMST inference reference. Never overwrite old fixtures.
beta <- c(.2,.35,-.1)
V <- matrix(c(.04,.012,-.008,.012,.03,.006,-.008,.006,.02),3,3)
e <- eigen(V,symmetric=TRUE)
root <- e$vectors %*% diag(sqrt(e$values)) %*% t(e$vectors)
z <- matrix(qnorm((((1:120)*7) %% 113+.5)/113),40,3,byrow=TRUE)
draws <- sweep(z %*% root,2,beta,"+")
x <- c(-.5,.25,.8)
offset <- cbind(.03*x,-.02*x)
entry <- c(0,.5,1)

eta <- function(b,family) {
    first <- b[1]+b[2]*x
    cbind(first,if(family=="piecewise_exponential") first+b[3] else b[3])+offset
}
logsf <- function(t,p,family) {
    if(family=="piecewise_exponential") return(-exp(p[1])*pmin(t,1)-exp(p[2])*pmax(0,t-1))
    if(family=="weibull") return(pweibull(t,shape=exp(p[2]),scale=exp(p[1]),lower.tail=FALSE,log.p=TRUE))
    plnorm(t,meanlog=p[1],sdlog=exp(p[2]),lower.tail=FALSE,log.p=TRUE)
}
functional <- function(b,family,kind,grid) {
    p <- eta(b,family)
    result <- matrix(NaN,length(x),length(grid))
    for(i in seq_along(x)) for(j in seq_along(grid)) if(grid[j]>=entry[i]) {
        t <- grid[j]
        if(kind=="hazard") {
            result[i,j] <- if(family=="piecewise_exponential") exp(p[i,if(t<=1) 1 else 2]) else
                exp((if(family=="weibull") dweibull(t,shape=exp(p[i,2]),scale=exp(p[i,1]),log=TRUE) else
                    dlnorm(t,meanlog=p[i,1],sdlog=exp(p[i,2]),log=TRUE))-logsf(t,p[i,],family))
        } else if(t==entry[i]) result[i,j] <- 0 else {
            breaks <- c(entry[i],if(family=="piecewise_exponential" && entry[i]<1 && t>1) 1,t)
            result[i,j] <- sum(vapply(seq_len(length(breaks)-1),function(k)
                integrate(function(u) exp(logsf(u,p[i,],family)-logsf(entry[i],p[i,],family)),
                          breaks[k],breaks[k+1],rel.tol=1e-11,abs.tol=1e-12)$value,0.))
        }
    }
    result
}
out <- list()
for(family in c("weibull","lognormal","piecewise_exponential")) for(kind in c("hazard","rmst")) {
    grid <- if(kind=="hazard") c(.2,.5,1,2) else c(0,.5,1,2.5)
    valid <- outer(entry,grid,"<=")
    point <- functional(beta,family,kind,grid)
    samples <- simplify2array(lapply(seq_len(nrow(draws)),function(i) functional(draws[i,],family,kind,grid)))
    center <- if(kind=="hazard") log(point) else point
    values <- if(kind=="hazard") log(samples) else samples
    se <- apply(values,c(1,2),sd)
    active <- valid & se>0
    maxima <- vapply(seq_len(nrow(draws)),function(i) max(abs((values[,,i]-center)[active]/se[active])),0.)
    critical <- unname(quantile(maxima,.95,type=8))
    lower <- center-critical*se
    upper <- center+critical*se
    if(kind=="hazard") { lower <- exp(lower); upper <- exp(upper) } else {
        lower <- pmax(lower,0)
        upper <- pmin(upper,outer(entry,grid,function(a,t) t-a))
    }
    limits <- apply(samples,c(1,2),quantile,probs=c(.025,.975),na.rm=TRUE,type=7)
    for(i in seq_along(x)) for(j in seq_along(grid)) out[[length(out)+1]] <- data.frame(
        family=family,kind=kind,subject=i-1,time=grid[j],estimate=point[i,j],
        lower=limits[1,i,j],upper=limits[2,i,j],sim_lower=lower[i,j],sim_upper=upper[i,j],
        se=se[i,j],critical=critical)
}
path <- "tests/reference_fixtures/survival_functional_inference.csv"
if(file.exists(path)) stop("Reference already exists; this script does not rebaseline it.")
write.csv(do.call(rbind,out),path,row.names=FALSE,na="NaN")
cat("Wrote new independent R reference:",path,"\n")
