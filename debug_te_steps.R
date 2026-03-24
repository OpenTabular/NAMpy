suppressPackageStartupMessages(library(mgcv))

set.seed(7)
n <- 80
x0 <- runif(n)
x1 <- runif(n)
y <- sin(2*pi*x0) * cos(2*pi*x1) + rnorm(n, sd=0.3)
data <- data.frame(x0=x0, x1=x1, y=y)

# Build the te smooth manually following mgcv's smooth.construct.tensor.smooth.spec
# Step 1: Build marginals with smooth.construct (no scale.penalty, no constraints)
spec0 <- s(x0, k=5, bs="cr")
spec1 <- s(x1, k=5, bs="cr")
sm0 <- smooth.construct(spec0, data, knots=NULL)
sm1 <- smooth.construct(spec1, data, knots=NULL)

X0_raw <- sm0$X
S0_raw <- sm0$S[[1]]
X1_raw <- sm1$X
S1_raw <- sm1$S[[1]]

cat("X0_raw shape:", nrow(X0_raw), ncol(X0_raw), "\n")
cat("S0_raw[1,1]:", S0_raw[1,1], "\n")

# Step 2: np reparam
# For marginal 0
np0 <- ncol(X0_raw)
knt0 <- seq(min(x0), max(x0), length=np0)
pd0 <- data.frame(x0=knt0)
names(pd0) <- sm0$term
sv0 <- svd(Predict.matrix(sm0, pd0))
cat("sv0$d:", sv0$d, "\n")
XP0 <- sv0$v %*% (t(sv0$u) / sv0$d)

X0_np <- X0_raw %*% XP0
S0_np <- t(XP0) %*% S0_raw %*% XP0
cat("X0_np[1,1]:", X0_np[1,1], "\n")
cat("S0_np[1,1]:", S0_np[1,1], "\n")

# For marginal 1
np1 <- ncol(X1_raw)
knt1 <- seq(min(x1), max(x1), length=np1)
pd1 <- data.frame(x1=knt1)
names(pd1) <- sm1$term
sv1 <- svd(Predict.matrix(sm1, pd1))
XP1 <- sv1$v %*% (t(sv1$u) / sv1$d)

X1_np <- X1_raw %*% XP1
S1_np <- t(XP1) %*% S1_raw %*% XP1

# Step 3: Eigenvalue normalize
ev0 <- eigen(S0_np, symmetric=TRUE, only.values=TRUE)$values[1]
ev1 <- eigen(S1_np, symmetric=TRUE, only.values=TRUE)$values[1]
cat("ev0 (max eig S0_np):", ev0, "\n")
cat("ev1 (max eig S1_np):", ev1, "\n")
S0_norm <- S0_np / ev0
S1_norm <- S1_np / ev1

# Step 4: Tensor product
X_te <- tensor.prod.model.matrix(list(X0_np, X1_np))
S_te_list <- tensor.prod.penalties(list(S0_norm, S1_norm))
cat("X_te shape:", nrow(X_te), ncol(X_te), "\n")
cat("S_te_list[[1]][1,1]:", S_te_list[[1]][1,1], "\n")
cat("S_te_list[[2]][1,1]:", S_te_list[[2]][1,1], "\n")

# Step 5: Outer scale.penalty
maXX <- norm(X_te, type="I")^2
cat("maXX:", maXX, "\n")
for (i in 1:length(S_te_list)) {
  maS <- norm(S_te_list[[i]]) / maXX
  S_te_list[[i]] <- S_te_list[[i]] / maS
}
cat("After scale.penalty S_te[1][1,1]:", S_te_list[[1]][1,1], "\n")
cat("After scale.penalty S_te[2][1,1]:", S_te_list[[2]][1,1], "\n")

# Step 6: Constraint absorption
C <- matrix(colMeans(X_te), 1, ncol(X_te))
cat("C[1,1:3]:", C[1,1:3], "\n")

k <- ncol(X_te)
j <- nrow(C)
qrc <- qr(t(C))
for (i in 1:length(S_te_list)) {
  ZSZ <- qr.qty(qrc, S_te_list[[i]])[(j+1):k,]
  S_te_list[[i]] <- t(qr.qty(qrc, t(ZSZ))[(j+1):k,])
}
X_te_c <- t(qr.qty(qrc, t(X_te))[(j+1):k,])

cat("After constraint X_te_c shape:", nrow(X_te_c), ncol(X_te_c), "\n")
cat("Final S_te[1][1,1]:", S_te_list[[1]][1,1], "\n")
cat("Final S_te[2][1,1]:", S_te_list[[2]][1,1], "\n")

# Verify against smoothCon
sm_te <- smoothCon(te(x0, x1, bs=c("cr","cr"), k=c(5,5)), data, absorb.cons=TRUE, scale.penalty=TRUE)[[1]]
cat("\nsmoothCon S[1][1,1]:", sm_te$S[[1]][1,1], "\n")
cat("smoothCon S[2][1,1]:", sm_te$S[[2]][1,1], "\n")
cat("smoothCon X[1,1]:", sm_te$X[1,1], "\n")
