Basis functions, penalties and identifiability
==============================================

A smooth is a function space together with a rule for measuring roughness.
Choosing a basis determines which functions the model can represent. Choosing
a penalty determines which of those functions it prefers when data are weak.
The individual :doc:`basis pages <index>` explain their particular construction;
this page develops the shared mathematics.

From a curve to a design matrix
-------------------------------

Represent a function by k known basis functions and unknown coefficients:

.. math::

   f(x)=\sum_{r=1}^{k}b_r(x)\beta_r,\qquad B_{ir}=b_r(x_i).

For n observed rows, the vector of contributions is B beta. The basis can
contain local spline pieces, radial kernels, graph eigenvectors, group
indicators or products of marginal functions. Individual coefficients usually
have no standalone effect-size interpretation: the curve B beta is the object
of interest. A change of basis can change every coefficient while leaving the
fitted function unchanged.

Each smooth contributes a block to the full design matrix. Parametric columns
and an intercept join those blocks, while a known offset o is added separately:

.. math::

   \eta=X\theta+o.

The compiled coefficient width p is the sum of retained block widths, not the
number of input variables. Tensor products can make p grow quickly: marginal
widths 6 and 5 give 30 raw product columns before constraints.

Roughness becomes a quadratic form
----------------------------------

For a cubic smooth, substituting the basis into integrated squared curvature
gives

.. math::

   \int\{f''(x)\}^2dx
   =\sum_{r,s}\beta_r\beta_s\int b_r''(x)b_s''(x)dx
   =\beta^\mathsf TS\beta.

S is positive semidefinite: roughness is nonnegative, but some nonzero
functions can have zero penalty. Difference penalties replace derivative
integrals with differences of coefficient sequences. Graph penalties measure
differences between adjacent regions. Several matrices can describe separate
directions of a tensor or spatially varying adaptive roughness:

.. math::

   S_\lambda=\sum_{j=1}^{s}\lambda_jS_j,\qquad
   \mathcal L(\theta)=-\ell(\theta)+\tfrac12\theta^\mathsf TS_\lambda\theta.

The coefficient estimate minimizes this penalized negative log likelihood.
Lambda trades fit against roughness; k limits the approximation space. If k
is too small, reducing lambda cannot create missing basis directions. If k
is generous, the penalty can still produce a simple fitted curve.

Null spaces, centering and term removal
---------------------------------------

In an eigendecomposition S=U diag(d) U-transpose, zero eigenvalues identify
unpenalized directions. A second-derivative penalty has constant and linear
null-space functions. As lambda grows, the penalized directions vanish and
the fitted curve approaches this null space. It need not approach zero.

An intercept and a constant smooth direction would be indistinguishable.
Ordinary smooths typically impose a constraint such as a zero training-row
sum. If C beta=0 and Z spans its null space, set beta=Z alpha:

.. math::

   B_* = BZ,\qquad S_*=Z^\mathsf TSZ.

Both the design and penalty must transform together. This removes ambiguity
without changing the allowed identified functions. A point constraint instead
anchors a function value; factor-by terms and random effects have their own
rules. Centering is a coordinate convention, not evidence of zero population
effect. The fitted constraint is retained when evaluating new data.

Shrinkage bases such as ts/cs penalize the old null space weakly. A separate
selection penalty estimates its strength independently. Either construction
can permit a whole term to approach zero; ordinary roughness penalization
alone generally cannot.

Effective degrees of freedom
----------------------------

For Gaussian identity-link data with diagonal prior precision W, a convenient
local view is

.. math::

   A_\lambda=B(B^\mathsf TW B+S_\lambda)^{-1}B^\mathsf TW,
   \qquad \mathrm{EDF}=\operatorname{tr}(A_\lambda).

Use an identified design and a rank-aware inverse in singular cases. EDF
measures fitted sensitivity, rather than counting basis coefficients. In
non-Gaussian models a corresponding influence matrix uses converged working
weights. Increasing lambda usually reduces flexibility, but multiple penalties
and nonlinear constraints make the whole fitted system the relevant object.
See :doc:`../inference` for NAMpy's reported EDF and covariance conventions.

Geometry and scientific assumptions
-----------------------------------

Isotropic kernels measure joint distances; their meaning depends on coordinate
units. Tensor products permit separate penalties along different units.
Cyclic bases encode a known period, soap bases encode boundaries, and MRF
bases encode adjacency. These choices supply structural information that
automatic smoothing selection cannot discover from an unrelated basis.

SCOP shape bases additionally transform free coefficients into a restricted
set. Their function is nonlinear in those free parameters, even though it is
linear in the transformed spline coefficients. The derivative chain rule and
penalty coordinates must remain consistent during fitting.

Further reading
---------------

* Wood (2003), `Thin plate regression splines
  <https://doi.org/10.1111/1467-9868.00374>`_.
* Eilers and Marx (1996), `Flexible smoothing with B-splines and penalties
  <https://doi.org/10.1214/ss/1038425655>`_.
* Wood, Scheipl and Faraway (2013), `Straightforward intermediate rank tensor
  product smoothing in mixed models
  <https://doi.org/10.1007/s11222-012-9314-z>`_.
* Pya and Wood (2015), `Shape constrained additive models
  <https://doi.org/10.1007/s11222-013-9448-7>`_.
