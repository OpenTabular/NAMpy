if(Sys.getenv("NAMPY_REFRESH_REFERENCE_FIXTURES")!="1") stop("Set explicit refresh flag.")
path <- "tests/reference_fixtures/qgam_adaptive_bandwidth.csv"
if(file.exists(path)) stop("Reference exists; refusing to overwrite.")
.libPaths(c(".cache/mgcv-lib",.libPaths()))
library(qgam)
data <- read.csv("tests/reference_fixtures/qgam_calfast_auto_nonlinear_input.csv")
fit <- gam(y~s(x,k=4)+z,data=data,method="REML")
error <- getFromNamespace(".getErrParam","qgam")(.3,fit,fit$sig2)
d <- 2+sum(unique(pen.edf(fit)))
write.csv(data.frame(residual=(fit$y-fit$fitted.values)/sqrt(fit$sig2),variance=fit$sig2,
                    dimension=d,bandwidth=error*sqrt(2*pi*fit$sig2)/(2*log(2))),path,row.names=FALSE)
