# Regenerate with generate_neighbor_references.py; executes upstream sources.
args <- commandArgs(TRUE)
.libPaths(c(args[3], .libPaths()))
library(jsonlite)
library(mgcv)
library(survival)
input <- fromJSON(args[1], simplifyMatrix=TRUE)
root <- args[2]
source(file.path(root, "expectreg/R/asyregpen.lsfit.R"))
source(file.path(root, "qgam/R/log1pexp.R"))
source(file.path(root, "qgam/R/elf.R"))
`%||%` <- function(x,y) if(is.null(x)) y else x
source(file.path(root, "gratia/R/simultaneous-intervals.R"))
X <- input$design; y <- input$response; S <- input$penalty; w <- input$weights
p <- ncol(X); n <- nrow(X)
expectiles <- lapply(input$levels, function(tau) {
  # expectreg minimizes sum(asym*r^2) + b'P'P b. NAMpy uses b'Sb/2.
  fit <- asyregpen.lsfit(y*sqrt(w), X*sqrt(w), tau, lambda=.5,
    DD=diag(sqrt(diag(S))), nb=p, constmat=matrix(0,1,p))
  list(level=tau, coefficients=unname(fit$a))
})
elfs <- lapply(input$levels, function(tau) {
  family <- elf(theta=0, qu=tau, co=input$bandwidth)
  fn <- function(b) .5*sum(family$dev.resids(y,drop(X%*%b),w)) + .5*sum(b*(S%*%b))
  gr <- function(b) drop(.5*crossprod(X,family$Dd(y,drop(X%*%b),theta=0,wt=w,level=0)$Dmu)+S%*%b)
  fit <- optim(rep(0,p), fn, gr, method="BFGS", control=list(reltol=1e-13, maxit=10000))
  stopifnot(fit$convergence==0)
  list(level=tau, coefficients=fit$par, gradient=max(abs(gr(fit$par))))
})
quantiles <- lapply(input$levels, function(tau) {
  fit <- quantreg::rq.fit(X*w,y*w,tau=tau,method="br")
  list(level=tau, coefficients=unname(fit$coefficients))
})
penalized_quantiles <- lapply(input$levels,function(tau) {
  # Independent upstream QP solution: tiny numerical ridge makes quadprog's
  # strictly-PD requirement valid without materially changing the target.
  epsilon <- 1e-9
  D <- diag(epsilon,p+2*n); D[seq_len(p),seq_len(p)] <- S+diag(epsilon,p)
  objective <- c(rep(0,p),w*tau,w*(1-tau))
  equality <- cbind(X,diag(n),-diag(n))
  positive <- cbind(matrix(0,2*n,p),diag(2*n))
  fit <- quadprog::solve.QP(D,-objective,t(rbind(equality,positive)),c(y,rep(0,2*n)),meq=n)
  list(level=tau,coefficients=fit$solution[seq_len(p)], numerical_ridge=epsilon)
})
H <- input$history
cb <- dlnm::crossbasis(H,lag=c(0,ncol(H)-1),
  argvar=list(fun="poly",degree=2,scale=1,intercept=FALSE),
  arglag=list(fun="lin",intercept=TRUE))
series_cb <- dlnm::crossbasis(input$series,lag=2, group=input$groups,
  argvar=list(fun="lin"),arglag=list(fun="lin",intercept=TRUE))
L <- input$functional; V <- input$covariance; b <- input$beta
se <- sqrt(rowSums((L%*%V)*L)); estimate <- drop(L%*%b)
band <- simultaneous_intervals(estimate,se,L,V,n_sim=nrow(input$deviations),deviations=input$deviations)
# pammtools delegates its splitting to survival::survSplit; use that exact
# kernel for independent validation without installing its plotting stack.
surv <- as.data.frame(input$survival)
split <- survival::survSplit(Surv(entry,time,event) ~ .,data=surv,
  cut=input$cuts[-c(1,length(input$cuts))],start="tstart",end="time",event="event")
split$exposure <- split$time-split$tstart
split$offset <- log(split$exposure)
family_fits <- lapply(c("student_lss","beta_lss","nb_lss"),function(name) {
  xx <- cbind(1,input$families$x)
  yy <- input$families[[name]]
  likelihood <- function(b) {
    a <- drop(xx %*% b[1:2]); s <- exp(drop(xx %*% b[3:4]))
    switch(name,
      student_lss=gamlss.dist::dTF(yy,mu=a,sigma=s,nu=5,log=TRUE),
      beta_lss=gamlss.dist::dBE(yy,mu=plogis(a),sigma=1/sqrt(1+s),log=TRUE),
      nb_lss=gamlss.dist::dNBI(yy,mu=exp(a),sigma=s,log=TRUE))
  }
  fit <- optim(c(.3,.6,if(name=="beta_lss") 2 else -.5,0),function(b)-sum(likelihood(b)),
    method="BFGS",control=list(reltol=1e-13,maxit=10000))
  stopifnot(fit$convergence==0)
  list(name=name,coefficients=fit$par,log_density=likelihood(fit$par),objective=fit$value)
})
boundary <- input$nb_boundary
# dNBI intentionally substitutes a Poisson approximation below sigma=1e-4.
# Compare the exact NB2 likelihood with stats::dnbinom, and record both.
nb_boundary <- list(exact=dnbinom(boundary$y,mu=boundary$mu,size=1/boundary$alpha,log=TRUE),
                    gamlss_approximation=gamlss.dist::dNBI(boundary$y,mu=boundary$mu,sigma=boundary$alpha,log=TRUE))
design_data <- as.data.frame(input$design_data)
design_fit <- gam(y~s(x,bs="cr",k=6)+offset(off),data=design_data,sp=.3,method="REML")
newdata <- transform(design_data,x=x+.01,off=off+3)
sm <- design_fit$smooth[[1]]
bs <- coef(design_fit)[sm$first.para:sm$last.para]
design_reference <- list(fit=unname(predict(design_fit,type="link")),
  newfit=unname(predict(design_fit,newdata=newdata,type="link")),
  penalty=drop(.3*crossprod(bs,sm$S[[1]]%*%bs)),edf=sum(design_fit$edf))
result <- list(expectiles=expectiles,elf=elfs,quantiles=quantiles,nb_boundary=nb_boundary,design_reference=design_reference,
  penalized_quantiles=penalized_quantiles,cross_basis=unclass(cb),series_cross_basis=unclass(series_cb),
  band=band,band_se=se,band_estimate=estimate,survival_split=split,family_fits=family_fits,
  versions=list(R=R.version.string,mgcv=as.character(packageVersion("mgcv")),
    dlnm=as.character(packageVersion("dlnm")),quantreg=as.character(packageVersion("quantreg")),
    quadprog=as.character(packageVersion("quadprog")),gamlss.dist=as.character(packageVersion("gamlss.dist"))))
write_json(result,args[4],auto_unbox=TRUE,digits=16,na="null",pretty=TRUE)
