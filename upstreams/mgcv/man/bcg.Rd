\name{bcg}
\alias{bcg}
%- Also NEED an `\alias' for EACH other topic documented here.
\title{Censored Box-Cox Gaussian family}
\description{Family for use with \code{\link{gam}} or \code{\link{bam}}, implementing regression for censored
data that can be modelled as Gaussian after Box-Cox transformation. If \eqn{y>0}{y>0} is the response and
\deqn{z= \left \{ \begin{array}{ll} (y^\lambda-1)/\lambda & \lambda \ne 0 \\ \log(y) & \lambda =0
\end{array} \right . }{z = log(y) if lambda=0 and (y^lambda-1)/lambda otherwise.}
with mean \eqn{\mu}{m} and standard deviation \eqn{w^{-1/2}\exp(\theta)}{w^{-1/2}exp(theta)},
then \eqn{w^{1/2}(z-\mu)\exp(-\theta)}{w^{1/2}(z-m)exp(-theta)} follows an \eqn{N(0,1)}{N(0,1)} distribution. That is
\deqn{z \sim N(\mu,e^{2\theta}w^{-1}).}{z ~ N(m,exp(2 theta)/w).} \eqn{\theta}{theta} is a single scalar for all observations, and similarly Box-Cox parameter \eqn{\lambda}{lambda} is a single scalar. Observations may be left, interval or right censored or uncensored.

Note that the regression model here specifies the mean of the Box-Cox transformed response, not the mean of the response itself: this is rather different to the usual GLM approach. 
}

\usage{
bcg(theta=NULL,link="identity")
}
\arguments{
\item{theta}{ a 2-vector containing the Box-Cox parameter \eqn{\lambda}{lambda} and the log standard deviation parameter \eqn{\theta}{theta}. If supplied and positive then taken as a fixed value of \eqn{\lambda}{lambda} and \eqn{\exp(\theta)}{exp(theta)}. If supplied and second element negative taken as initial value and negative of initial value respectively.}
\item{link}{The link function: \code{"identity"}, \code{"log"} or \code{"sqrt"}.}
}
\value{
An object of class \code{extended.family}.
}

\details{If the family is used with a vector response, then it is assumed that there is no censoring. If there is censoring then the response should be supplied as a two column matrix. The first column is always numeric. Entries in the second column are as follows.
\itemize{
\item If an entry is identical to the corresponding first column entry, then it is an uncensored observation.
\item If an entry is numeric and different to the first column entry then there is interval censoring. The first column entry is the lower interval limit and the second column entry is the upper interval limit. \eqn{y}{y} is only known to be between these limits.
\item If the second column entry is \code{0} then the observation is left censored at the value of the entry in the first column. It is only known that \eqn{y}{y} is less than or equal to the first column value.
\item If the second column entry is \code{Inf} then the observation is right censored at the value of the entry in the first column. It is only known that \eqn{y}{y} is greater than or equal to the first column value.
}
Any mixture of censored and uncensored data is allowed, but be aware that data consisting only of right and/or left censored data contain very little information.
}

%- maybe also `usage' for other objects documented here.

\author{ Simon N. Wood \email{simon.wood@r-project.org}
}

\seealso{\code{\link{cnorm}}, \code{\link{cpois}}, \code{\link{clog}} }

\references{
Wood, S.N., N. Pya and B. Saefken (2016), Smoothing parameter and
model selection for general smooth models.
Journal of the American Statistical Association 111, 1548-1575
\doi{10.1080/01621459.2016.1180986}
}


\examples{
library(mgcv)

set.seed(3) ## Simulate some gamma data?
dat <- gamSim(1,n=400,dist="normal",scale=1)
dat$f <- dat$f/4 ## true linear predictor 
Ey <- exp(dat$f);scale <- .5 ## mean and GLM scale parameter
dat$y <- rgamma(Ey*0,shape=1/scale,scale=Ey*scale)

## Just Box-Cox no censoring...
b <- gam(y~s(x0)+s(x1)+s(x2)+s(x3),family=bcg,data=dat)
summary(b)
plot(b,pages=1,scheme=1)

## try various censoring...
yb <- cbind(dat$y,dat$y)
ind <- 1:100
yb[ind,2] <- yb[ind,1] + runif(100)*3
yb[51:100,2] <- 0 ## left censored
yb[101:140,2] <- Inf ## right censored

b <- gam(yb~s(x0)+s(x1)+s(x2)+s(x3),family=bcg,data=dat)
summary(b)
plot(b,pages=1,scheme=1)

}
\keyword{models} \keyword{regression}%-- one or more ..


