\name{dppois}
\alias{dppois}
%- Also NEED an `\alias' for EACH other topic documented here.
\title{Stable evaluation of difference between Poisson c.d.f.s}
\description{Evaluates the difference between two Poisson cumulative distribution functions avoiding cancellation error.  
}
\usage{
dppois(y0,y1,mu,log.p=TRUE)
}
%- maybe also `usage' for other objects documented here.
\arguments{
 \item{y0}{vector of lower values at which to evaluate Poisson distribution function.}
 \item{y1}{vector of upper values at which to evaluate Poisson distribution function.}
 \item{mu}{vector giving Poisson means}
 \item{log.p}{set to \code{FALSE} to compute on probability scale.}
}

\details{ Equivalent to \code{log(ppois(y1,mu)-ppois(y0,mu))}, but stable when \code{y0} and \code{y1} are highly improbable and in the same tail of the distribution (avoids cancellation to \code{log(0)}). }
 

\author{ Simon N. Wood \email{simon.wood@r-project.org}}

\examples{
require(mgcv)
n <- 10
mu <- rep(2,n)
y0 <- c(0:5,10,23,0,2)
y1 <- c(1:6,11,1000,100,40)
dppois(y0,y1,mu)
log(ppois(y1,mu)-ppois(y0,mu))
}

\keyword{models} \keyword{smooth} \keyword{regression}%-- one or more ..









