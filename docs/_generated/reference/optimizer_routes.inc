.. list-table:: Representative runtime-checked routes
   :header-rows: 1
   :class: api-reference-table

   * - Model
     - Family
     - Criterion
     - Outer optimizer
     - Coefficient solver
     - Correlation
     - Result
   * - Ordinary
     - gaussian
     - gcv
     - outer_newton
     - newton
     - Independent
     - Finite fit
   * - Ordinary
     - gaussian
     - reml
     - outer_newton
     - newton
     - Independent
     - Finite fit
   * - Ordinary
     - gaussian
     - reml
     - efs
     - newton
     - Independent
     - Finite fit
   * - Ordinary
     - gaussian
     - reml
     - optim
     - newton
     - Independent
     - Finite fit
   * - SCAM
     - gaussian
     - gcv
     - bfgs
     - newton
     - Independent
     - Finite fit
   * - SCAM
     - gaussian
     - gcv
     - efs
     - newton
     - Independent
     - Finite fit
   * - SCAM
     - gaussian
     - gcv
     - optim
     - newton
     - Independent
     - Finite fit
   * - SCAM
     - gaussian
     - gcv
     - nlm
     - newton
     - Independent
     - Finite fit
   * - SCAM
     - gaussian
     - gcv
     - nlm.fd
     - newton
     - Independent
     - Finite fit
   * - SCAM
     - gaussian
     - gcv
     - efs
     - bfgs
     - Independent
     - Finite fit
   * - SCAM
     - gaussian
     - reml
     - outer_newton
     - newton
     - Independent
     - Rejected as intended
   * - Transformed LSS
     - gaulss
     - fixed
     - outer_newton
     - newton
     - Independent
     - Finite fit
   * - Transformed LSS
     - gaulss
     - reml
     - outer_newton
     - newton
     - Independent
     - Rejected as intended
   * - Ordinary
     - gaussian
     - reml
     - outer_newton
     - newton
     - AR(1)
     - Finite fit
   * - Ordinary
     - poisson
     - reml
     - outer_newton
     - newton
     - AR(1)
     - Rejected as intended
