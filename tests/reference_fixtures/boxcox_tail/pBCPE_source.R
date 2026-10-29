# Installed gamlss.dist pBCPE probability implementation
pBCPE <-
function (q, mu = 5, sigma = 0.1, nu = 1, tau = 2, lower.tail = TRUE, 
    log.p = FALSE) 
{
    F.T <- function(t, tau) {
        log.c <- 0.5 * (-(2/tau) * log(2) + lgamma(1/tau) - lgamma(3/tau))
        c <- exp(log.c)
        s <- 0.5 * ((abs(t/c))^tau)
        F.s <- pgamma(s, shape = 1/tau, scale = 1)
        cdf <- 0.5 * (1 + F.s * sign(t))
        cdf
    }
    if (any(mu < 0)) 
        stop(paste("mu must be positive", "\n", ""))
    if (any(sigma < 0)) 
        stop(paste("sigma must be positive", "\n", ""))
    if (any(tau < 0)) 
        stop(paste("tau must be positive", "\n", ""))
    if (length(nu) > 1) 
        z <- ifelse(nu != 0, (((q/mu)^nu - 1)/(nu * sigma)), 
            log(q/mu)/sigma)
    else if (nu != 0) 
        z <- (((q/mu)^nu - 1)/(nu * sigma))
    else z <- log(q/mu)/sigma
    FYy1 <- F.T(z, tau)
    if (length(nu) > 1) 
        FYy2 <- ifelse(nu > 0, F.T(-1/(sigma * abs(nu)), tau), 
            0)
    else if (nu > 0) 
        FYy2 <- F.T(-1/(sigma * abs(nu)), tau)
    else FYy2 <- 0
    FYy3 <- F.T(1/(sigma * abs(nu)), tau)
    FYy <- (FYy1 - FYy2)/FYy3
    if (lower.tail == TRUE) 
        FYy <- FYy
    else FYy <- 1 - FYy
    if (log.p == FALSE) 
        FYy <- FYy
    else FYy <- log(FYy)
    FYy
}
