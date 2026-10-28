# Freeze only invariant numerical outputs, never raw basis columns/coefficients.
for (key in c('MGCV_LIB_PATH', 'SCAM_LIB_PATH')) {
  path <- Sys.getenv(key)
  if (nzchar(path)) .libPaths(c(path, .libPaths()))
}
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
# The repository's recorded local R library has a documented Newton-sign
# mutation (tests/reference_fixtures/manifest.json). Restore CRAN 1.9-4 in this
# process, as the fitted-likelihood and weighted-AR probes do. Never alter the
# installed package or treat its version string alone as source identity.
newton_ref <- get('newton', asNamespace('mgcv'))
body(newton_ref) <- parse(text=sub('Sstep <- grad/max', 'Sstep <- -grad/max',
    paste(deparse(body(newton_ref)), collapse='\n'), fixed=TRUE))[[1]]
assignInNamespace('newton', newton_ref, ns='mgcv')
a <- commandArgs(TRUE)
p <- fromJSON(a[1], simplifyVector=FALSE)
d <- as.data.frame(lapply(p$data, unlist))
nd <- as.data.frame(lapply(p$newdata, unlist))
if ('bound' %in% names(d)) d$bound <- as.numeric(d$bound)
for (nm in names(d)) if (is.character(d[[nm]])) {
  d[[nm]] <- factor(d[[nm]])
  if (nm %in% names(nd)) nd[[nm]] <- factor(nd[[nm]], levels=levels(d[[nm]]))
}
form <- lapply(unlist(p$formula), as.formula)
if (length(form) == 1) form <- form[[1]]
fam <- eval(parse(text=p$family))
if (identical(p$name, 'twlss')) {
  form <- list(y~s(x,bs='cr',k=6,sp=.7),~1,~s(x,bs='cr',k=5,sp=.9))
}
source_warnings <- character()
fit_source <- function(fun,args) withCallingHandlers(do.call(fun,args),warning=function(w) {
  source_warnings <<- c(source_warnings,conditionMessage(w))
})
args <- list(formula=form, data=d, family=fam, weights=d$w)
if (!is.null(p$knots)) args$knots <- lapply(p$knots, unlist)
if (!is.null(p$scale)) args$scale <- p$scale
if (identical(p$engine, 'scam')) {
  suppressPackageStartupMessages(library(scam))
  args$sp <- unlist(p$sp)
  args$start <- unlist(p$start)
  args$optimizer <- c('bfgs','newton')
  args$control <- list(maxit=300, devtol.fit=1e-9, steptol.fit=1e-9)
  b <- fit_source(scam, args)
  covariance <- b$Vp.t
  covariance_freq <- b$Ve.t
} else {
  args$method <- 'REML'
  args$control <- gam.control(epsilon=1e-10,newton=list(conv.tol=1e-10))
  if (identical(p$name, 'twlss')) args$optimizer <- 'efs'
  G <- do.call(gam, c(args,list(fit=FALSE)))
  # Setup may mutate extended-family censoring metadata. A fresh family is
  # essential: otherwise fitting reuses already-initialized response closures.
  args$family <- eval(parse(text=p$family))
  if (!identical(p$name, 'twlss') && length(G$sp)) args$sp <- rep(unlist(p$sp),length.out=length(G$sp))
  b <- fit_source(gam,args)
  covariance <- b$Vp
  covariance_freq <- b$Ve
}
covariance_unconditional <- b$Vc
# estimate.gam maps Vp and Ve through G$P but leaves Vc in fit coordinates.
# Evaluate every kernel in prediction coordinates, including fixed-SP t2.
if (!identical(p$engine,'scam') && !is.null(G$P) && !is.null(covariance_unconditional))
  covariance_unconditional <- G$P %*% covariance_unconditional %*% t(G$P)

# Stack each predictor's evaluation operator, retaining cross-predictor
# covariances while eliminating all dependence on the coefficient basis.
evaluation <- function(data) {
  if (identical(p$name,'lipl')) {
    # Native predict.scam SE multiplies unreduced lipl columns by reduced Vp.t.
    # Rebuild its documented identified prediction map from the R constructor.
    sm <- b$smooth[[1]]
    P <- cbind(1,mgcv::PredictMat(sm,data)[,2:sm$q1,drop=FALSE])
  } else P <- predict(b,data,type='lpmatrix')
  lpi <- attr(P,'lpi')
  if (is.null(lpi)) return(P)
  do.call(rbind,lapply(lpi,function(ii) {
    Z <- P*0; Z[,ii] <- P[,ii,drop=FALSE]; Z
  }))
}
E <- evaluation(nd)
X <- evaluation(d)
projector <- function(X) {
  sv <- svd(X,nu=min(dim(X)),nv=0)
  U <- sv$u[,sv$d > max(dim(X))*.Machine$double.eps*max(sv$d),drop=FALSE]
  tcrossprod(U)
}
if (identical(p$name,'lipl')) {
  eta_new <- drop(E %*% b$coefficients.t)+nd$off
  pr <- list(fit=eta_new,se.fit=sqrt(pmax(0,rowSums((E %*% covariance)*E))))
  response <- eta_new
} else {
  pr <- predict(b,nd,type='link',se.fit=TRUE)
  response <- predict(b,nd,type='response')
}
safe <- function(fun) tryCatch(unname(fun()),error=function(e) NULL)
ll <- safe(function() as.numeric(logLik(b)))
write_json(list(
  case_name=p$name, engine=p$engine, version=as.character(packageVersion(p$engine)),
  upstream_notes=if(identical(p$name,'lipl'))
    'Native predict.scam SE has a dimension error; constructor and fitted covariance evaluate the identified prediction kernel.' else
    if(!identical(p$engine,'scam') && !is.null(G$P) && !is.null(b$Vc))
      'Source Vc is in fit coordinates; G$P maps it into prediction coordinates before evaluating the invariant kernel.' else NULL,
  source_warnings=source_warnings, inner_converged=b$converged,
  outer_status=if(is.null(b$outer.info)) NULL else b$outer.info$conv,
  edf=sum(b$edf), edf1=if(is.null(b$edf1)) NULL else sum(b$edf1), aic=safe(function() AIC(b)), bic=safe(function() BIC(b)),
  loglik=ll, likelihood_df=safe(function() attr(logLik(b),'df')),
  deviance=unname(deviance(b)), scale=unname(b$sig2),
  eta=unname(pr$fit), se=unname(pr$se.fit),
  fitted=unname(predict(b,type='link')),
  response=unname(response),
  covariance_kernel=unname(E %*% covariance %*% t(E)),
  frequentist_kernel=if(is.null(covariance_freq)) NULL else unname(E %*% covariance_freq %*% t(E)),
  unconditional_kernel=if(is.null(covariance_unconditional)) NULL else unname(E %*% covariance_unconditional %*% t(E)),
  theta=if(is.function(b$family$getTheta)) safe(function() {
    if(length(formals(b$family$getTheta))) b$family$getTheta(TRUE) else b$family$getTheta()
  }) else NULL,
  design_projector=unname(projector(X)),
  residuals=list(response=safe(function() residuals(b,type='response')),
                 pearson=safe(function() residuals(b,type='pearson')),
                 deviance=safe(function() residuals(b,type='deviance')))
),a[2],digits=17,auto_unbox=TRUE,null='null',na='null')
