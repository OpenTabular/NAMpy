## deprecated code only still here because of reverse dependencies

gam.fit <- function (G, start = NULL, etastart = NULL, 
    mustart = NULL, family = gaussian(), 
    control = gam.control(),gamma=1,
    fixedSteps=(control$maxit+1),...)
## deprecated since 1.9-0 (July 2023), but still imported by refund, simml etc
# fitting function for a gam, modified from glm.fit.
# note that smoothing parameter estimates from one irls iterate are carried over to the next irls iterate
# unless the range of s.p.s is large enough that numerical problems might be encountered (want to avoid 
# completely flat parts of gcv/ubre score). In the latter case autoinitialization is requested.
# fixedSteps < its default causes at most fixedSteps iterations to be taken,
# without warning if convergence has not been achieved. This is useful for
# obtaining starting values for outer iteration.
{   .Deprecated(msg="Internal mgcv function gam.fit is no longer used - see gam.fit3.")
    intercept<-G$intercept
    conv <- FALSE
    n <- nobs <- NROW(G$y) ## n just there to keep codetools happy
    nvars <- NCOL(G$X) # check this needed
    y<-G$y # original data
    X<-G$X # original design matrix
    if (nvars == 0) stop("Model seems to contain no terms")
    olm <- G$am   # only need 1 iteration as it's a pure additive model.
    if (!olm)  .Deprecated(msg="performance iteration with gam is deprecated, use bam instead")
    find.theta<-FALSE # any supplied -ve binomial theta treated as known, G$sig2 is scale parameter
    if (substr(family$family[1],1,17)=="Negative Binomial")
    { Theta <- family$getTheta()
      if (length(Theta)==1) { ## Theta fixed
        find.theta <- FALSE
        G$sig2 <- 1
      } else {
        if (length(Theta)>2)
        warning("Discrete Theta search not available with performance iteration")
        Theta <- range(Theta)
        T.max <- Theta[2]          ## upper search limit
        T.min <- Theta[1]          ## lower search limit
        Theta <- sqrt(T.max*T.min) ## initial value
        find.theta <- TRUE
      }
      nb.link<-family$link # negative.binomial family, there's a choise of links
    }

    
    # obtain average element sizes for the penalties
    n.S<-length(G$S)
    if (n.S>0)
    { S.size<-0
      for (i in 1:n.S) S.size[i]<-mean(abs(G$S[[i]])) 
    }
    weights<-G$w # original weights

    n.score <- sum(weights!=0) ## n to use in GCV score (i.e. don't count points with no influence)   

    offset<-G$offset 

    variance <- family$variance;dev.resids <- family$dev.resids
    aic <- family$aic
    linkinv <- family$linkinv;linkfun <- family$linkfun;mu.eta <- family$mu.eta
    if (!is.function(variance) || !is.function(linkinv)) 
        stop("illegal `family' argument")
    valideta <- family$valideta
    if (is.null(valideta)) 
        valideta <- function(eta) TRUE
    validmu <- family$validmu
    if (is.null(validmu)) 
        validmu <- function(mu) TRUE
    if (is.null(mustart))   # new from version 1.5.0 
    { eval(family$initialize)}
    else 
    { mukeep <- mustart
      eval(family$initialize)
      mustart <- mukeep
    }
    if (NCOL(y) > 1) 
        stop("y must be univariate unless binomial")
    
    coefold <- NULL                 # 1.5.0
    eta <- if (!is.null(etastart))  # 1.5.0
        etastart

    else if (!is.null(start)) 
    if (length(start) != nvars) 
    stop(gettextf("Length of start should equal %d and correspond to initial coefs.",nvars)) 
    else 
    { coefold<-start                        #1.5.0
      offset+as.vector(if (NCOL(G$X) == 1)
       G$X * start
       else G$X %*% start)
    }
    else family$linkfun(mustart)
    mu <- linkinv(eta)
    if (!(validmu(mu) && valideta(eta))) 
        stop("Can't find valid starting values: please specify some")
    devold <- sum(dev.resids(y, mu, weights))
   
    boundary <- FALSE
    scale <- G$sig2

    msp <- G$sp
    magic.control<-list(tol=G$conv.tol,step.half=G$max.half,#maxit=control$maxit+control$globit,
                          rank.tol=control$rank.tol)

    for (iter in 1:(control$maxit)) 
    {
        good <- weights > 0
        varmu <- variance(mu)[good]
        if (any(is.na(varmu))) 
            stop("NAs in V(mu)")
        if (any(varmu == 0)) 
            stop("0s in V(mu)")
        mu.eta.val <- mu.eta(eta)
        if (any(is.na(mu.eta.val[good]))) 
            stop("NAs in d(mu)/d(eta)")
        good <- (weights > 0) & (mu.eta.val != 0) # note good modified here => must re-calc each iter
        if (all(!good)) {
            conv <- FALSE
            warning(gettextf("No observations informative at iteration %d", 
                iter))
            break
        }
   
        mevg <- mu.eta.val[good];mug <- mu[good];yg <- y[good]
        weg <- weights[good];##etag <- eta[good]
        var.mug<-variance(mug)

        G$y <- z <- (eta - offset)[good] + (yg - mug)/mevg
        w <- sqrt((weg * mevg^2)/var.mug)
        
        G$w<-w
        G$X<-X[good,,drop=FALSE]  # truncated design matrix       
     
        # must set G$sig2 to scale parameter or -1 here....
        G$sig2 <- scale


        if (sum(!is.finite(G$y))+sum(!is.finite(G$w))>0) 
        stop("iterative weights or data non-finite in gam.fit - regularization may help. See ?gam.control.")

        ## solve the working weighted penalized LS problem ...

        mr <- magic(G$y,G$X,msp,G$S,G$off,L=G$L,lsp0=G$lsp0,G$rank,G$H,matrix(0,0,ncol(G$X)),  #G$C,
                    G$w,gamma=gamma,G$sig2,G$sig2<0,
                    ridge.parameter=control$irls.reg,control=magic.control,n.score=n.score,nthreads=control$nthreads)
        G$p<-mr$b;msp<-mr$sp;G$sig2<-mr$scale;G$gcv.ubre<-mr$score;

        if (find.theta) {# then family is negative binomial with unknown theta - estimate it here from G$sig2
            ##  need to get edf array
          mv <- magic.post.proc(G$X,mr,w=G$w^2)
          G$edf <- mv$edf

          Theta<-mgcv.find.theta(Theta,T.max,T.min,weights,good,mu,mu.eta.val,G,.Machine$double.eps^0.5)
          family<-do.call("negbin",list(theta=Theta,link=nb.link))
          variance <- family$variance;dev.resids <- family$dev.resids
          aic <- family$aic
          family$Theta <- Theta ## save Theta estimate in family
        }

        if (any(!is.finite(G$p))) {
            conv <- FALSE   
            warning(gettextf("Non-finite coefficients at iteration %d",iter))
            break
        }

		
        start <- G$p
        eta <- drop(X %*% start) # 1.5.0
        mu <- linkinv(eta <- eta + offset)
        eta <- linkfun(mu) # force eta/mu consistency even if linkinv truncates
        dev <- sum(dev.resids(y, mu, weights))
        if (control$trace) 
            message(gettextf("Deviance = %s Iterations - %d", dev, iter, domain = "R-mgcv"))
        boundary <- FALSE
        if (!is.finite(dev)) {
            if (is.null(coefold))
            stop("no valid set of coefficients has been found:please supply starting values",
            call. = FALSE)
            warning("Step size truncated due to divergence",call.=FALSE)
            ii <- 1
            while (!is.finite(dev)) {
                if (ii > control$maxit) 
                  stop("inner loop 1; can't correct step size")
                ii <- ii + 1
                start <- (start + coefold)/2
                eta<-drop(X %*% start)
                mu <- linkinv(eta <- eta + offset)
                eta <- linkfun(mu) 
                dev <- sum(dev.resids(y, mu, weights))
            }
            boundary <- TRUE
            if (control$trace) 
                cat("Step halved: new deviance =", dev, "\n")
        }
        if (!(valideta(eta) && validmu(mu))) {
            warning("Step size truncated: out of bounds.",call.=FALSE)
            ii <- 1
            while (!(valideta(eta) && validmu(mu))) {
                if (ii > control$maxit) 
                  stop("inner loop 2; can't correct step size")
                ii <- ii + 1
                start <- (start + coefold)/2
                eta<-drop(X %*% start)
                mu <- linkinv(eta <- eta + offset)
                eta<-linkfun(mu)
            }
            boundary <- TRUE
            dev <- sum(dev.resids(y, mu, weights))
            if (control$trace) 
                cat("Step halved: new deviance =", dev, "\n")
        }

        ## Test for convergence here ...

        if (abs(dev - devold)/(0.1 + abs(dev)) < control$epsilon || olm ||
            iter >= fixedSteps) {
            conv <- TRUE
            coef <- start #1.5.0
            break
        }
        else {
            devold <- dev
            coefold <- coef<-start
        }
    }
    if (!conv) 
    { warning("Algorithm did not converge") 
    }
    if (boundary) 
        warning("Algorithm stopped at boundary value")
    eps <- 10 * .Machine$double.eps
    if (family$family[1] == "binomial") {
        if (any(mu > 1 - eps) || any(mu < eps)) 
            warning("fitted probabilities numerically 0 or 1 occurred")
    }
    if (family$family[1] == "poisson") {
        if (any(mu < eps)) 
            warning("fitted rates numerically 0 occurred")
    }
    
    residuals <- rep(NA, nobs)
    residuals[good] <- z - (eta - offset)[good]
    
    wt <- rep(0, nobs)
    wt[good] <- w^2
   
    wtdmu <- if (intercept) sum(weights * y)/sum(weights) else linkinv(offset)
    nulldev <- sum(dev.resids(y, wtdmu, weights))
    n.ok <- nobs - sum(weights == 0)
    nulldf <- n.ok - as.integer(intercept)
    
    ## Extract a little more information from the fit....

    mv <- magic.post.proc(G$X,mr,w=G$w^2)
    G$Vp<-mv$Vb;G$hat<-mv$hat;
    G$Ve <- mv$Ve # frequentist cov. matrix
    G$edf<-mv$edf
    G$conv<-mr$gcv.info
    G$sp<-msp
    rank<-G$conv$rank

    ## use MLE of scale in Gaussian case - best estimate otherwise. 
    dev1 <- if (scale>0) scale*sum(weights) else if (family$family=="gaussian") dev else G$sig2*sum(weights)  

    aic.model <- aic(y, n, mu, weights, dev1) + 2 * sum(G$edf)
    if (scale < 0) { ## deviance based GCV
      gcv.ubre.dev <- n.score*dev/(n.score-gamma*sum(G$edf))^2
    } else { # deviance based UBRE, which is just AIC
      gcv.ubre.dev <- dev/n.score + 2 * gamma * sum(G$edf)/n.score - G$sig2
    }
  
	list(coefficients = as.vector(coef), residuals = residuals, fitted.values = mu, 
        family = family,linear.predictors = eta, deviance = dev,
        null.deviance = nulldev, iter = iter, weights = wt, prior.weights = weights,  
        df.null = nulldf, y = y, converged = conv,sig2=G$sig2,edf=G$edf,edf1=mv$edf1,hat=G$hat,
        ##F=mv$F,
        R=mr$R,
        boundary = boundary,sp = G$sp,nsdf=G$nsdf,Ve=G$Ve,Vp=G$Vp,rV=mr$rV,mgcv.conv=G$conv,
        gcv.ubre=G$gcv.ubre,aic=aic.model,rank=rank,gcv.ubre.dev=gcv.ubre.dev,scale.estimated = (scale < 0))
} ## gam.fit
