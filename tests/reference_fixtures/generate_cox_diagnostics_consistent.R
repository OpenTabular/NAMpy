# Source baseline/linear-predictor identity, isolating mgcv stratified indexing.
if(Sys.getenv("NAMPY_REFRESH_REFERENCE_FIXTURES")!="1") stop("Set explicit reference refresh flag.")
path <- "tests/reference_fixtures/cox_diagnostics_consistent.csv"
if(file.exists(path)) stop("Reference exists; refusing to overwrite.")
library(mgcv)
data <- read.csv("tests/reference_fixtures/cox_fixed_input.csv")
data$response <- cbind(data$time,data$stratum)
fit <- gam(response~x+z,data=data,weights=event,offset=offset,family=cox.ph(),method="REML",control=gam.control(epsilon=1e-10))
# postproc uses local r in global h for stratified fitted values; predict() also
# sorts query rows without sorting varying offsets. Capture the source hazard
# identity directly, with each original row's own stratum, time and offset.
fd <- fit$family$data
H <- vapply(seq_len(nrow(data)),function(i) {
  index <- which(fd$tr.strat==data$stratum[i]&fd$tr<=data$time[i])
  fd$h[index[which.max(fd$tr[index])]]*exp(fit$linear.predictors[i])
},numeric(1))
M <- data$event-H
D <- sign(M)*sqrt(-2*(M+data$event*log(pmax(H,1e-50))))
write.csv(data.frame(row=seq_len(nrow(data)),martingale=M,deviance=D),path,row.names=FALSE)
