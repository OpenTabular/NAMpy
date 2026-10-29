# mgcv 1.9-4 coxlpl/coxpp at fixed predictors: event/censor ties, no fitting.
if (Sys.getenv("NAMPY_REFRESH_REFERENCE_FIXTURES") != "1") stop("Set explicit fixture refresh flag.")
path <- "tests/reference_fixtures/neural_cox.csv"
if (file.exists(path)) stop("Reference exists; refusing to overwrite.")
library(mgcv)
time <- c(1,1,2,3,2,4,4,1)
event <- c(1,0,1,1,0,1,0,1)
eta <- c(.1,-.4,.6,.2,-.1,.8,-.3,.5)
ix <- order(time,decreasing=TRUE)
t <- time[ix]; e <- event[ix]; b <- eta[ix]; X <- diag(length(t))
f <- cox.ph()
l <- f$ll(t, X, b, e, f, offset=rep(0,length(t)), deriv=1)
tr <- unique(t); r <- match(t,tr); nt <- length(tr)
o <- .C("coxpp",as.double(b),A=as.double(X),as.integer(r),d=as.integer(e),
        h=double(nt),q=double(nt),km=double(nt),n=as.integer(length(t)),
        p=as.integer(ncol(X)),nt=as.integer(nt),PACKAGE="mgcv")
# Restore subject order; baseline values use each subject's observed time.
write.csv(data.frame(time,event,eta,log_likelihood=l$l,
                    gradient=l$lb[order(ix)],baseline=o$h[match(time,tr)],
                    mgcv_version=as.character(packageVersion("mgcv"))),path,row.names=FALSE)
