# mgcv and SCAM optimizer coverage

NAMpy ports the following smoothing-selection routes from the vendored mgcv
1.9-4 and SCAM sources:

- GCV, GACV, UBRE/Cp, ML, REML, P-ML, P-REML, LAML, NCV, and QNCV;
- outer Newton, BFGS, EFS, `optim`, `nlm`, and Gaussian performance iteration
  (`magic`);
- known-scale Gaussian and Gamma UBRE/Cp through `GAM(scale=...)`;
- `gam.control`-style solver settings through `gam_control()`;
- Tweedie EFS and `min_sp` handling;
- SCAM EFS, `optim`, `nlm`, `nlm.fd`, and coefficient BFGS; and
- the Gaussian identity, unknown-scale AR(1) ML/REML/LAML determinant
  correction from `bam`.

The Python implementation uses SciPy for the R `optim`/`nlm` numerical engine.
It uses the upstream objective, analytic derivatives, bounds, and mapped
tolerance controls while preserving the public optimizer identity. It does not
claim identical stopping codes or iteration-by-iteration traces from R's
numerical routines.

## Intentional upstream boundaries

These combinations remain guarded rather than approximated:

- SCAM automatic ML/REML/LAML selection. Upstream SCAM does not supply the
  combined transformed-coefficient likelihood route; SCAM GCV/UBRE remains
  supported.
- Automatic smoothing for transformed models with multiple linear predictors.
  Fixed smoothing is supported, but the required transformed Laplace
  derivatives are not supplied by the upstream combined model.
- `nlm.fd` for ordinary mgcv GAMs. This spelling is exposed by SCAM only.
- `magic` outside Gaussian GCV/UBRE/Cp performance iteration.
- AR(1) likelihood selection outside Gaussian identity models with unknown
  scale. The supported route uses the `bam` determinant correction, not a GCV
  response-transform heuristic.
- NCV/QNCV for general or extended families, and the optional post-fit
  jackknife pass when `nei=` is used with another criterion. The ordinary GLM
  neighborhood-deletion route is implemented.

Controls without a Python or native-kernel analogue are retained as metadata or
documented no-ops. In particular, `nthreads` and `ncv_threads` do not create
parallel native kernels.
