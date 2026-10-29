# Independent R normalized likelihood, not mgcv cnorm's distinct semantics.
if(Sys.getenv('NAMPY_REFRESH_REFERENCE_FIXTURES')!='1') stop('Explicit refresh required')
paths <- paste0('tests/reference_fixtures/normal_observations',c('_input.csv','.csv'))
if(any(file.exists(paths))) stop('Refusing to overwrite references')
set.seed(718)
n <- 180; x <- seq(-1,1,length.out=n)
mu <- .25+.3*x; sigma <- .7
lo <- -.5; hi <- 1.7
u <- runif(n)
y <- qnorm(pnorm(lo,mu,sigma)+u*(pnorm(hi,mu,sigma)-pnorm(lo,mu,sigma)),mu,sigma)
lower <- upper <- y
lower[y < -.1] <- -Inf; upper[y < -.1] <- -.1
lower[y>1.2] <- 1.2; upper[y>1.2] <- Inf
interval <- y>=.3 & y<.5
lower[interval] <- .3; upper[interval] <- .5
objective <- function(b) {
  mean <- b[1]+b[2]*x; sd <- exp(b[3])
  exact <- lower==upper
  terms <- numeric(n)
  terms[exact] <- dnorm(lower[exact],mean[exact],sd,log=TRUE)
  terms[!exact] <- log(pnorm(pmin(upper[!exact],hi),mean[!exact],sd)-pnorm(pmax(lower[!exact],lo),mean[!exact],sd))
  -sum(terms-log(pnorm(hi,mean,sd)-pnorm(lo,mean,sd)))
}
fit <- optim(c(.2,.3,log(.7)),objective,method='BFGS',control=list(reltol=1e-13,maxit=1000))
if(fit$convergence!=0) stop('Reference failed')
write.csv(data.frame(x=x,lower=lower,upper=upper),paths[1],row.names=FALSE)
write.csv(data.frame(intercept=fit$par[1],slope=fit$par[2],log_scale=fit$par[3],nll=fit$value,R=as.character(getRversion())),paths[2],row.names=FALSE)
