# Independent response-space quadrature of installed GAMLSS densities.
if(Sys.getenv('NAMPY_REFRESH_REFERENCE_FIXTURES')!='1') stop('Explicit refresh required')
output <- 'tests/reference_fixtures/boxcox_moments.csv'
if(file.exists(output)) stop('Refusing to overwrite reference')
lib <- Sys.getenv('GAMLSS_LIB_PATH',unset='')
if(nzchar(lib)) .libPaths(c(lib,.libPaths()))
library(gamlss.dist)
rows <- list()
for (family in c('BCT','BCPE')) for (nu in c(-3,-2.5,.5,1,2)) {
  tau <- if(family=='BCT') 12 else 2.5
  mu <- 1.4; sigma <- .4
  d <- get(paste0('d',family))
  first <- integrate(function(y)y*d(y,mu,sigma,nu,tau),0,Inf,rel.tol=1e-10,subdivisions=1000)$value
  second <- integrate(function(y)y^2*d(y,mu,sigma,nu,tau),0,Inf,rel.tol=1e-10,subdivisions=1000)$value
  rows[[length(rows)+1]] <- data.frame(family=tolower(family),mu=mu,sigma=sigma,nu=nu,tau=tau,mean=first,variance=second-first^2)
}
write.csv(do.call(rbind,rows),output,row.names=FALSE)
