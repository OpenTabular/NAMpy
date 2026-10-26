library(mgcv)
library(jsonlite)
args <- commandArgs(TRUE)
p <- fromJSON(args[1])
d <- as.data.frame(p$data)
d$y <- d[[paste0("y_", p$family)]]
d$B <- I(as.matrix(d[, c("x", "u", "v")]))
PP <- list(B=list(diag(c(1,0,0)),diag(c(0,1,0)),diag(c(0,0,1)),
                  L=as.matrix(p$L),sp=unlist(p$sp)))
family <- if (p$family == "gaussian") gaussian() else poisson()
b <- gam(y~B,data=d,family=family,method="REML",paraPen=PP,
         control=gam.control(epsilon=1e-10,newton=list(conv.tol=1e-10)))
invisible(capture.output(v <- gam.vcomp(b,rescale=FALSE)))
out <- list(names=rownames(v$vc),sd=unname(v$vc[,1]),
            all_names=names(v$all),all=unname(v$all))
write_json(out,args[2],digits=17,auto_unbox=TRUE,null="null",na="null")
