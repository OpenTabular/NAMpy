Main Divergences

  1. High: t2 numeric by prediction uses training by, not new-data by.
     nampy/gam/smooths/tensor/t2.py:398 returns _apply_cached_by(...); that helper uses fit-time values from nampy/gam/smooths/smooth_base.py:512.
     te/ti correctly use new data. This can break or mis-scale predict() for t2(..., by=z).
  2. High: factor level handling loses mgcv factor semantics.
     nampy/gam/smooths/categorical/categorical_utils.py:48 uses np.unique(observed_values). mgcv uses levels(fac) in fs, sz, mrf, and model.matrix
     for re. NAMpy drops unused categorical levels, loses custom category order, and treats numeric pandas categoricals as numeric. Affects column
     order, penalty order, prediction zeros, linked parity.
  3. High: tensor d= grouping missing.
     mgcv te/t2 support d to group multiple covariates into one marginal and coerce invalid multivariate bases to tp (mgcv/R/smooth.r:398-440). NAMpy
     tensor code assumes one feature per marginal in nampy/gam/smooths/tensor/marginals.py:221. Any mgcv formula like te(x,z,w, d=[2,1],
     bs=["tp","cr"]) has no faithful smooths-layer analogue.
  4. Medium: tensor m normalization ambiguous vs mgcv list/vector semantics.
     mgcv: scalar m repeats; flat vector length n.bases gives one scalar per margin; list gives per-margin vectors. NAMpy nampy/gam/smooths/tensor/
     marginals.py:138 cannot distinguish Python flat list intended as one margin’s vector from per-margin scalars. This can mis-specify ps/tp tensor
     marginals.
  5. Medium: t2 sign hacks are not upstream logic.
     nampy/gam/smooths/algebra.py:138 flips columns by basis name using local observed conventions. mgcv just uses eigen(..., symmetric=TRUE) inside
     nat.param(type=3). This is representation-parity fragile across LAPACK/R/eigen sign behavior.
  6. Medium: tp/ts invalid low m behavior differs.
     NAMpy raises when 2*m <= d in nampy/splines/univariate/tp.py:187. mgcv resets to default order in mgcv/src/tprs.c:168 and mgcv/src/tprs.c:568.
     Smooths calling ThinPlateSplineTerm(..., m=1) in 2D/3D can reject where mgcv silently resets.
  7. Medium: tp/gp large-data knot subsampling not mgcv RNG.
     NAMpy uses NumPy default_rng in nampy/splines/univariate/tp.py:292 and nampy/splines/univariate/gp.py:225. mgcv uses R sample() under
     temp.seed() (mgcv/R/smooth.r:1290-1301, 3495-3498). For xt["max.knots"] paths, selected knots differ, so basis differs.
  8. Low/medium: direct runtime t2.transform_new() violates local BaseSmoothTerm contract.
     Base contract says transform_new emits fitted coefficient parameterization. t2.fit() stores constrained basis_train after full_transform, but
     transform_new() emits raw and relies on compiler predict_coefficient_map. Compiled path may be okay; direct smooth runtime is inconsistent.

  Good Matches Seen
  cr/cs/cc, ps, te/ti main control flow mostly mirrors mgcv: raw marginal penalties, np transform, eigen normalization, tensor penalty order, outer
  scaling, and constraint placement look intentionally ported. re, mrf, fs, sz also mirror major block assembly, but factor-level semantics above are
  big remaining parity risk.

  Parity Status
  No tests run. Parity likely holds for covered common numeric cases, but remains vulnerable for t2 by=, pandas categorical/factor order, tensor d=,
  low-order tp m, and large-data tp/gp subsampling.