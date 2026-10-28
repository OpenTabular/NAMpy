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
     - Coefficient stopping status
     - Outer stopping status
     - Warnings
     - Rejection reason
   * - Ordinary
     - gaussian
     - gcv
     - outer_newton
     - newton
     - Independent
     - Finite fit
     - Converged (1 iterations)
     - Converged (4 iterations): full convergence
     - None captured
     - Not applicable
   * - Ordinary
     - gaussian
     - reml
     - outer_newton
     - newton
     - Independent
     - Finite fit
     - Converged (1 iterations)
     - Converged (4 iterations): full convergence
     - None captured
     - Not applicable
   * - Ordinary
     - gaussian
     - reml
     - efs
     - newton
     - Independent
     - Finite fit
     - Converged (1 iterations)
     - Converged (4 iterations): full convergence
     - None captured
     - Not applicable
   * - Ordinary
     - gaussian
     - reml
     - optim
     - newton
     - Independent
     - Finite fit
     - Converged (1 iterations)
     - Converged (8 iterations): 0
     - None captured
     - Not applicable
   * - SCAM
     - gaussian
     - gcv
     - bfgs
     - newton
     - Independent
     - Finite fit
     - Converged (3 iterations)
     - Converged (3 iterations): Full convergence
     - None captured
     - Not applicable
   * - SCAM
     - gaussian
     - gcv
     - efs
     - newton
     - Independent
     - Finite fit
     - Converged (3 iterations)
     - Converged (4 iterations): full convergence
     - None captured
     - Not applicable
   * - SCAM
     - gaussian
     - gcv
     - optim
     - newton
     - Independent
     - Finite fit
     - Converged (3 iterations)
     - Converged (17 iterations): Optimization terminated successfully.
     - None captured
     - Not applicable
   * - SCAM
     - gaussian
     - gcv
     - nlm
     - newton
     - Independent
     - Finite fit
     - Converged (3 iterations)
     - Converged (5 iterations): Relative gradient is close to zero, current iterate is probably solution
     - None captured
     - Not applicable
   * - SCAM
     - gaussian
     - gcv
     - nlm.fd
     - newton
     - Independent
     - Finite fit
     - Converged (3 iterations)
     - Converged (5 iterations): Relative gradient is close to zero, current iterate is probably solution
     - None captured
     - Not applicable
   * - SCAM
     - gaussian
     - gcv
     - efs
     - bfgs
     - Independent
     - Finite fit
     - Converged (25 iterations)
     - Converged (4 iterations): full convergence
     - None captured
     - Not applicable
   * - SCAM
     - gaussian
     - reml
     - outer_newton
     - newton
     - Independent
     - Rejected as intended
     - Not run
     - Not run
     - None captured
     - Transformed-coefficient smoothing selection currently supports GCV/UBRE only for the single-predictor GLM kernel.
   * - Transformed LSS
     - gaulss
     - fixed
     - Not applicable
     - newton
     - Independent
     - Finite fit
     - Converged (8 iterations)
     - Not applicable
     - None captured
     - Not applicable
   * - Transformed LSS
     - gaulss
     - reml
     - outer_newton
     - newton
     - Independent
     - Rejected as intended
     - Not run
     - Not run
     - None captured
     - Automatic smoothing for transformed multi-predictor/LSS models requires higher-order transformed Laplace derivatives that are not implemented. Fixed smoothing is supported.
   * - Ordinary
     - gaussian
     - reml
     - outer_newton
     - newton
     - AR(1)
     - Finite fit
     - Converged (1 iterations)
     - Converged (3 iterations): full convergence
     - None captured
     - Not applicable
   * - Ordinary
     - poisson
     - reml
     - outer_newton
     - newton
     - AR(1)
     - Rejected as intended
     - Not run
     - Not run
     - None captured
     - Automatic smoothing selection with method='reml' and an observation transform is outside the determinant-corrected Gaussian bam route. Use fixed smoothing or GCV.
