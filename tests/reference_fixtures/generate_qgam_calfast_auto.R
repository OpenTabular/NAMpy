# Independent qgam 2.0.1 fixed-co, automatically selected SP calibration.
if(Sys.getenv("NAMPY_REFRESH_REFERENCE_FIXTURES")!="1") stop("Set explicit refresh flag.")
args <- commandArgs(trailingOnly=TRUE)
k <- if(length(args)) as.integer(args[1]) else 4
prefix <- paste0("tests/reference_fixtures/qgam_calfast_auto",if(k==4) "" else paste0("_k",k))
paths <- paste0(prefix,c("_input.csv","_profile.csv",".csv"))
if(any(file.exists(paths))) stop("Reference exists; refusing to overwrite.")
.libPaths(c(".cache/mgcv-lib",.libPaths()))
library(qgam)
n <- 100
i <- 0:(n-1)
data <- data.frame(x=sin(i*.43),z=cos(i*.29))
data$y <- .4+.6*data$x+.25*sin(i*1.71)+.4*cos(i*.83)
co <- .2; qu <- .3; sp <- .5
# Use the public ELF family to hold co and sp fixed. This isolates loss-scale
# calibration from qgam's separate Gaussian/shash bandwidth selection.
criterion <- function(log_sigma, details=FALSE) {
 fit <- gam(as.formula(sprintf("y~s(x,k=%d)+z",k)),data=data,family=elf(theta=log_sigma,qu=qu,co=co),
            method="REML",control=gam.control(epsilon=1e-10))
 X <- predict(fit,type="lpmatrix")
 P <- matrix(0,ncol(X),ncol(X))
 first <- fit$smooth[[1]]$first.para; last <- fit$smooth[[1]]$last.para
 P[first:last,first:last] <- fit$sp[1]*fit$smooth[[1]]$S[[1]]
 p <- plogis((data$y-fit$fitted.values)/co)
 score <- (p-1+qu)/exp(log_sigma)
 OFI <- crossprod(sqrt(p*(1-p)/(co*exp(log_sigma)))*X)
 G <- score*X
 EXXT <- crossprod(X)/n
 EXEXT <- tcrossprod(colMeans(X))
 ESS <- sum(abs(score))^2/sum(score^2)
 alpha <- min(ESS/ncol(X)^2,1)
 Vsim <- EXXT*mean(score^2)-mean(score)^2*EXEXT
 V <- n*(alpha*cov(G)+(1-alpha)*Vsim)
 inv <- getFromNamespace(".invert_psdef_matrix","qgam")
 VS <- inv(OFI%*%inv(V,fit$rank)%*%OFI+P,fit$rank)
 varPost <- rowSums((X%*%fit$Vp)*X)
 varSand <- rowSums((X%*%VS)*X)
 loss <- mean(sqrt(varSand/varPost+log(varPost/varSand)))
 if(!details) return(loss)
 list(fit=fit,X=X,loss=loss,VS=VS,varPost=varPost,varSand=varSand)
}
profile <- do.call(rbind,lapply(c(-2,-1,0,1),function(log_sigma) {
 current <- criterion(log_sigma,TRUE)
 data.frame(log_sigma,sp=current$fit$sp[1],loss=current$loss,subject=0:(n-1),fitted=current$fit$fitted.values,
            posterior_variance=current$varPost,sandwich_variance=current$varSand)
}))
opt <- optimize(criterion,c(-3,2),tol=1e-8)
final <- criterion(opt$minimum,TRUE)
endpoint <- data.frame(subject=0:(n-1),log_sigma=opt$minimum,sp=final$fit$sp[1],loss=final$loss,
 fitted=final$fit$fitted.values,posterior_variance=final$varPost,sandwich_variance=final$varSand)
write.csv(data,paths[1],row.names=FALSE)
write.csv(profile,paths[2],row.names=FALSE)
write.csv(endpoint,paths[3],row.names=FALSE)
