# Fixed-endpoint companion to near_singular_reml_endpoint_probe.py.
# It records mgcv's coefficient-level EDF attribution at the mgcv and NAMpy
# boundary endpoints without running either outer optimizer.

suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))

f <- factor(c("b", "a", "c", "a", "b", "c", "a", "c"))
effects <- c(a = 1.5, b = -0.25, c = 0.75)
d <- data.frame(y = unname(effects[as.character(f)]), f = f)

log_sp <- c(mgcv = -64.52515079189783, nampy = -70.47282965270232)
out <- lapply(log_sp, function(rho) {
  fit <- gam(y ~ s(f, bs = "re"), data = d, family = gaussian(),
             method = "REML", sp = exp(rho))
  list(
    log_sp = unname(rho),
    edf = unname(as.numeric(fit$edf)),
    smooth_edf = unname(sum(fit$edf[2:4])),
    trace_H = unname(sum(fit$edf)),
    deviance = unname(as.numeric(fit$deviance))
  )
})

write_json(out, stdout(), auto_unbox = TRUE, digits = 17, pretty = TRUE)
