if(Sys.getenv("NAMPY_REFRESH_REFERENCE_FIXTURES")!="1") stop("Set explicit reference refresh flag.")
paths <- paste0("tests/reference_fixtures/qgam_bandwidth",c("_input.csv",".csv"))
if(any(file.exists(paths))) stop("Reference exists; refusing to overwrite.")
.libPaths(c(".cache/mgcv-lib",.libPaths()))
library(qgam)
i <- 0:159
residual <- .4*sin(i*1.7)+.6*cos(i*.8)+.2*sin(i*.7)^2
# A minimal initial Gaussian fit record exercises the public source function's
# data contract; d=5 is independently supplied via a parametric design.
data <- data.frame(y=residual,x1=sin(i*.31),x2=cos(i*.91),x3=sin(i*.43),x4=cos(i*.23))
initial <- gam(y~x1+x2+x3+x4,data=data)
r <- as.numeric((initial$y-initial$fitted.values)/sqrt(initial$sig2))
fitshash <- getFromNamespace(".fitShash","qgam")(r)
levels <- c(.2,.5,.8)
error <- getFromNamespace(".getErrParam","qgam")(levels,initial,initial$sig2)
write.csv(data.frame(residual=r,variance=initial$sig2,dimension=5),paths[1],row.names=FALSE)
write.csv(data.frame(level=levels,error=error,bandwidth=error*sqrt(2*pi*initial$sig2)/(2*log(2)),
                     p1=fitshash$par[1],p2=fitshash$par[2],p3=fitshash$par[3],p4=fitshash$par[4]),paths[2],row.names=FALSE)
