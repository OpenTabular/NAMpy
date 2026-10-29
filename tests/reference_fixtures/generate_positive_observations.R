# Independent R dlnorm/plnorm and dweibull/pweibull likelihoods (R stats).
# Parameterizations match gamlss.dist/R/logNO.R and R/WEI.R.
if(Sys.getenv('NAMPY_REFRESH_REFERENCE_FIXTURES')!='1') stop('Explicit refresh required')
for(family in c('lognormal','weibull')) {
  paths <- paste0('tests/reference_fixtures/',family,'_observations',c('_input.csv','.csv'))
  if(any(file.exists(paths))) stop('Refusing to overwrite references')
  set.seed(718)
  n <- 180; x <- seq(-1,1,length.out=n)
  first <- .2+.3*x; second <- if(family=='lognormal') log(.7) else log(1.4)
  cdf <- if(family=='lognormal') function(y,a,b) plnorm(y,a,exp(b)) else function(y,a,b) pweibull(y,exp(b),exp(a))
  density <- if(family=='lognormal') function(y,a,b) dlnorm(y,a,exp(b),log=TRUE) else function(y,a,b) dweibull(y,exp(b),exp(a),log=TRUE)
  quantile <- if(family=='lognormal') function(p,a,b) qlnorm(p,a,exp(b)) else function(p,a,b) qweibull(p,exp(b),exp(a))
  lo <- .2; hi <- 3
  y <- quantile(cdf(lo,first,second)+runif(n)*(cdf(hi,first,second)-cdf(lo,first,second)),first,second)
  lower <- upper <- y
  lower[y < .5] <- -Inf; upper[y < .5] <- .5
  lower[y>2] <- 2; upper[y>2] <- Inf
  interval <- y>=.8 & y<1.1
  lower[interval] <- .8; upper[interval] <- 1.1
  objective <- function(b) {
    first <- b[1]+b[2]*x; second <- b[3]
    exact <- lower==upper
    terms <- numeric(n)
    terms[exact] <- density(lower[exact],first[exact],second)
    terms[!exact] <- log(cdf(pmin(upper[!exact],hi),first[!exact],second)-cdf(pmax(lower[!exact],lo),first[!exact],second))
    -sum(terms-log(cdf(hi,first,second)-cdf(lo,first,second)))
  }
  fit <- optim(c(.2,.3,second),objective,method='BFGS',control=list(reltol=1e-13,maxit=1000))
  if(fit$convergence!=0) stop('Reference failed')
  write.csv(data.frame(x=x,lower=lower,upper=upper),paths[1],row.names=FALSE)
  write.csv(data.frame(intercept=fit$par[1],slope=fit$par[2],second=fit$par[3],nll=fit$value,R=as.character(getRversion())),paths[2],row.names=FALSE)
}
