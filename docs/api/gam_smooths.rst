Smooth Terms
============

Smooth terms construct the basis and penalty matrices used by the statistical
GAM backend. Users normally select them by code inside a formula instead of
instantiating the runtime classes directly:

.. code-block:: python

   formula = (
       'y ~ s(x, bs="tp", k=12) '
       '+ s(day, bs="cc", k=10) '
       '+ ti(latitude, longitude, bs=c("cr", "cr"))'
   )

The inventory below lists every public basis code. For guidance based on the
modeling question and fitted plots, start with :ref:`formula-visual-reference`.
The classes at the end of this page form the extension API for custom smooths.

Formula constructors
--------------------

.. list-table::
   :header-rows: 1
   :widths: 16 46 38

   * - Constructor
     - Representation
     - Minimal formula
   * - ``s(...)``
     - One smooth term using the selected ``bs=`` code; some bases accept
       multiple variables.
     - ``y ~ s(x, bs="tp", k=12)``
   * - ``te(...)``
     - Full tensor product, including marginal directions.
     - ``y ~ te(x, z, bs=c("cr", "cr"))``
   * - ``ti(...)``
     - Tensor interaction with marginal main-effect spaces removed.
     - ``y ~ s(x) + s(z) + ti(x, z)``
   * - ``t2(...)``
     - Full tensor product with SS-ANOVA-style range/null-space penalty blocks.
     - ``y ~ t2(x, z, bs=c("cr", "cr"))``

See the :ref:`fitted tensor comparison <tensor-constructor-comparison>` before
choosing among ``te()``, ``ti()``, and ``t2()``.

.. _gam-smooth-bases:

Metric, cyclic, and spatial basis codes
---------------------------------------

Supply these codes through ``bs=`` inside ``s()``. Codes marked as tensor
marginals can also appear in the ``bs=c(...)`` argument of ``te()``, ``ti()``,
or ``t2()``.

.. list-table::
   :header-rows: 1
   :widths: 14 48 38

   * - Code
     - Basis
     - Minimal formula
   * - ``tp``
     - Thin-plate regression spline; accepts one or more covariates and can be
       a tensor marginal.
     - ``s(x, bs="tp")``
   * - ``ts``
     - Thin-plate spline with null-space shrinkage; accepts one or more
       covariates and can be a tensor marginal.
     - ``s(x, bs="ts")``
   * - ``cr``
     - Cubic regression spline; can be a tensor marginal.
     - ``s(x, bs="cr")``
   * - ``cs``
     - Cubic regression spline with shrinkage; can be a tensor marginal.
     - ``s(x, bs="cs")``
   * - ``cc``
     - Cyclic cubic regression spline; can be a tensor marginal.
     - ``s(hour, bs="cc")``
   * - ``ps``
     - P-spline; can be a tensor marginal.
     - ``s(x, bs="ps", m=c(2, 2))``
   * - ``cp``
     - Cyclic P-spline; can be a tensor marginal.
     - ``s(hour, bs="cp")``
   * - ``bs``
     - B-spline with derivative-based penalties; can be a tensor marginal.
     - ``s(x, bs="bs", m=c(3, 2))``
   * - ``ds``
     - Duchon spline for one or more covariates; can be a tensor marginal.
     - ``s(x, z, bs="ds")``
   * - ``gp``
     - Gaussian-process smooth for one or more covariates; can be a tensor
       marginal.
     - ``s(x, z, bs="gp")``
   * - ``ad``
     - Adaptive smooth for one or two covariates, with locally varying
       smoothness.
     - ``s(x, bs="ad")``
   * - ``sos``
     - Spline on a sphere from latitude and longitude; can be a tensor
       marginal.
     - ``s(lat, lon, bs="sos")``
   * - ``so`` / ``sf`` / ``sw``
     - Soap film on a bounded planar domain: full surface, boundary film,
       or zero-boundary interior component. Requires boundary loops and knots.
     - ``s(x, z, bs="so", xt={...})``
   * - ``sc`` / ``scad``
     - mgcv's B-spline/adaptive constructor branches without inequalities.
       Shape inequalities require the separate, unsupported ``scasm`` fitter.
     - ``s(x, bs="sc")``

.. _soap-film-smooths:

Soap films on bounded domains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Soap films respect irregular boundaries and holes. The ``so`` basis combines
a smooth boundary interpolant (``sf``) with an interior component (``sw``)
that is zero on the boundary. Each free boundary loop has its own penalty;
the interior has another penalty. ``k`` controls the boundary basis dimension,
while the number of paired interior knots controls the interior dimension.

Pass boundary loops through ``xt['bnd']``, with coordinate names matching the
formula. Each loop must describe a closed polygon; an omitted closing point is
added automatically. Supply interior knot coordinates through the model's
``knots`` argument. For example:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from nampy.gam import GAM

   rng = np.random.default_rng(42)
   data = pd.DataFrame(rng.uniform(0.05, 0.95, (150, 2)), columns=['x', 'z'])
   data['y'] = np.sin(4 * data.x) * np.cos(3 * data.z)
   boundary = [{'x': [0., 1., 1., 0., 0.],
                'z': [0., 0., 1., 1., 0.]}]
   gx, gz = np.meshgrid([0.2, 0.5, 0.8], [0.2, 0.5, 0.8])
   knots = {'x': gx.ravel(), 'z': gz.ravel()}
   xt = {'bnd': boundary, 'nmax': 40}
   model = GAM(
       formula=f"y ~ s(x, z, bs='so', k=8, xt={xt!r})",
       knots=knots, smoothing_method='reml', optimize_smoothing=True,
   ).fit(data=data)
   prediction, standard_error = model.predict(data, return_se=True)

Use additional loops to describe holes, and a list of ``k`` values to give
each loop a different boundary dimension. ``xt['nmax']`` controls the
finite-difference grid resolution (default 200); changing it changes the
numerical basis. ``xt['bndSpec']`` selects the cyclic boundary spline, using
``{'bs': 'cc'}`` by default or ``{'bs': 'cp', 'm': [2, 2]}``.

An optional ``f`` array on a boundary loop supplies known boundary values.
These become a fixed smooth-owned offset for ``so``/``sf``, including during
new-data prediction and component decomposition. They are added to any formula
or caller-supplied offset. ``sw`` has a zero boundary and does not use this
offset. A boundary-only ``sf`` term needs at least one free boundary loop.

Training observations outside the domain are rejected; prediction rows outside
the domain return NaNs. ``so`` drops outside knots with a warning, following
mgcv; ``sf`` and ``sw`` retain upstream knot validation. Single-penalty ``sf``
and ``sw`` terms can be margins of ``te``, ``ti`` and ``t2`` using ordinary
input columns. For example, ``te(x,z,t,bs=['sf','cr'],d=[2,1],k=[8,5],
xt=[{'bnd': boundary},None])`` combines a boundary film with a time smooth;
supply the soap knots as for a direct term. The full ``so`` basis and ``sf``
with multiple free boundary loops have multiple penalties and remain rejected
as tensor margins. With ordinary input columns, ``fs`` accepts a single-penalty
``sf`` base and ``sz`` accepts ``sf`` or ``sw``; for example,
``s(x, z, group, bs='fs', xt={'bs': 'sf', 'bnd': boundary})``.
The upstream factor-smooth wrappers discard the soap-owned boundary offset;
these nested terms follow that behavior even with known boundary segments.
Soap factor nesting with matrix coordinates and analytic soap derivatives
remain unavailable.
The static reference tests cover basis and penalty matrices, domains with
holes, known boundaries, fitting, prediction uncertainty, and persistence.

The newer ``sc``/``scad`` codes are distinct from SCAM's ``mpi`` and related
bases below. Their default, unconstrained branches reproduce ``bs`` and
adaptive P-splines respectively. Requests such as ``xt=['m+']`` raise
``NotImplementedError``: NAMpy does not run mgcv's separate inequality-based
``scasm`` fitter. Use the documented SCAM bases for supported constrained fits.

.. _structured-smooth-bases:

Structured basis codes
----------------------

.. list-table::
   :header-rows: 1
   :widths: 14 48 38

   * - Code
     - Structure
     - Minimal formula
   * - ``re``
     - Random effects or random coefficients; a single-penalty term can be a tensor marginal.
     - ``s(group, bs="re")``
   * - ``mrf``
     - Markov random field over a supplied neighborhood graph; can be a tensor
       marginal.
     - ``s(region, bs="mrf", xt=graph)``
   * - ``fs``
     - Factor-specific smooth curves with factor-smooth penalties.
     - ``s(x, group, bs="fs")``
   * - ``sz``
     - Sum-to-zero factor-smooth deviations around lower-order effects.
     - ``s(x, group, bs="sz")``

``by=`` modulates an ordinary smooth by a numeric or factor variable, while
``id=`` links smoothing parameters across compatible terms. They are smooth
options rather than basis codes.

.. _shape-constrained-basis-codes:

Shape-constrained basis codes
-----------------------------

These codes use the same ``s(..., bs=...)`` syntax. Their constraints,
coefficient transforms, and supported fitting routes are documented in
:doc:`../user_guide/shape_constrained_gams`.

.. list-table::
   :header-rows: 1
   :widths: 24 48 28

   * - Constraint
     - Codes
     - Minimal formula
   * - Monotone
     - ``mpi``, ``mpd``
     - ``s(x, bs="mpi")``
   * - Convex or concave
     - ``cx``, ``cv``
     - ``s(x, bs="cx")``
   * - Monotone with curvature
     - ``micx``, ``micv``, ``mdcx``, ``mdcv``
     - ``s(x, bs="micx")``
   * - Positive
     - ``po``, ``ipo``, ``dpo``, ``cpop``
     - ``s(x, bs="po")``
   * - Anchored monotone
     - ``miso``, ``mifo``
     - ``s(x, bs="miso")``
   * - Numeric-by shape constrained
     - ``mpiby``, ``mpdby``, ``micxby``, ``micvby``, ``mdcxby``,
       ``mdcvby``, ``cxby``, ``cvby``
     - ``s(x, by=z, bs="mpiby")``
   * - Local
     - ``lmpi``, ``lipl``
     - ``s(x, bs="lmpi", xt=list(xc=0.4))``
   * - Bivariate derivative constraints
     - ``tedmi``, ``tedmd``, ``temicx``, ``temicv``, ``tedecx``,
       ``tedecv``, ``tecxcx``, ``tecvcv``, ``tecxcv``
     - ``s(x, z, bs="tedmi")``
   * - Bivariate sum/axis constraints
     - ``tescx``, ``tescv``, ``tesmi1``, ``tesmd1``, ``tesmi2``,
       ``tesmd2``, ``tismi``, ``tismd``
     - ``s(x, z, bs="tesmi1")``

Base contract and registry
--------------------------

.. currentmodule:: nampy.gam.smooths

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BaseSmoothTerm
   ByState
   make_smooth_term
   register_smooth

One-dimensional and metric smooths
----------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ThinPlateSplineTerm
   CubicSplineTerm
   PSplineTerm1D
   DerivativeBSplineTerm1D
   DuchonSplineTerm
   GaussianProcessTerm
   AdaptiveSmoothTerm
   SphericalSplineTerm

Tensor and structured smooths
-----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   TensorProductSplineTerm
   InteractionTensorProductSplineTerm
   AlternativeTensorProductSplineTerm
   RandomEffectTerm
   MarkovRandomFieldTerm
   FSmoothInteractionTerm
   SZSmoothInteractionTerm
   ShapeConstrainedPSplineTerm

.. _matrix-functional-smooths:

Matrix-valued smooths
---------------------

``tp`` and ``ts`` accept matrix-valued covariates stored as array-valued
DataFrame cells. Each observation contributes a sum of pointwise smooth values:
``s(X, by=L, bs="tp")`` evaluates ``sum_j L[i,j] * f(X[i,j])``.
Omitting ``by`` uses unit weights. For a surface,
``s(X, Z, by=L, bs="ts")`` evaluates paired locations
``sum_j L[i,j] * f(X[i,j], Z[i,j])``; all coordinate and weight matrices must
have matching shapes.

The same weighted or unit-weight summation is supported by ``cr``,
``cc``, ``ps``, ``cp``, ``bs``, ``ds``, ``gp`` and ``sos``. Duchon and GP
terms accept paired multivariate locations; spherical terms use paired
latitude/longitude matrices. Penalties are scaled on the underlying pointwise
basis before summation. Constant row sums retain the upstream centering
constraint even when the individual quadrature weights vary.
Repeated coordinates follow mgcv's setup packing rule: coordinates are pooled
in column-major order and packed to their first occurrences when fewer than
90% of the point tuples are unique. Penalty scaling and centering use that
setup basis; prediction expands the pointwise basis before summation. All ``cs``
matrix fits are rejected, including distinct locations: unequal null-space
shrinkage penalties have an unresolved portable eigenvector-orientation parity
issue. Scalar ``cs`` terms remain supported.

Adaptive ``ad`` terms support weighted or unit-weight one-dimensional
matrices and unit-weight paired two-dimensional matrices. Weighted 2D
adaptive matrix fits are rejected: mgcv 1.9-4 loses the ``by`` variable in
prediction for this construction, so its fitted and predicted values differ.

Soap ``so``/``sf``/``sw`` terms also support paired location matrices and
weighted sums. Known-boundary ``so``/``sf`` matrix fitting is rejected because
upstream ``smoothCon`` cannot aggregate the constructor's boundary offset.
``sw`` has no boundary offset and supports this case.

Random-effect ``re`` and full/reduced-rank ``mrf`` terms accept array-valued
categorical cells as well. Their factor levels are pooled across all cells;
string labels such as ``"01"`` and ``"1"`` remain distinct.
Numeric and categorical random-effect interactions retain pointwise products
before summation. For example, ``s(region, by=L, bs='re')`` accepts a
``region`` column whose cells are lists of labels, while ``s(X, region,
by=L, bs='re')`` sums numeric-by-factor random effects at paired locations.
MRF terms retain their required adjacency or penalty specification and reject
unknown regions; random effects give unseen levels zero contribution.

``fs`` terms with ``xt={'bs': 'cr'}`` or ``xt={'bs': 'ps'}`` accept matrix
metric locations and a scalar factor per observation. They support weighted,
unit and constant-row-sum functionals, including repeated paired locations.
Matrix ``fs`` with its default ``tp`` base remains rejected because its
natural-parameter null-space orientation has unresolved parity. Specify an
accepted base explicitly, for example ``s(X, group, bs='fs', k=7,
xt={'bs': 'cr'}, by=L)``.

``sz`` matrix functionals support ``tp`` (one or two metric coordinates),
``cr`` and ``ps`` bases, with one or more scalar factor columns per observation.
Unit, varying and constant-row-sum weights follow upstream constraint handling;
repeated-location ``cr``/``ps`` cases are also validated. Matrix soap bases
inside either ``fs`` or ``sz`` remain unavailable.

``te``, ``ti`` and ``t2`` support paired matrix coordinates with tested
``cr``, ``ps``, ``tp``, ``ds``, ``gp``, ``sf`` and ``sw`` margins.
``te`` and ``ti`` additionally support ``bs`` margins.
Matrix ``t2`` terms with any ``bs`` margin are rejected because portable
null-space normalization parity remains unresolved. ``te``/``ti`` with ``bs``
and ordinary scalar-input ``t2`` remain supported; matrix ``t2`` with tested
``cr``, ``ds``, ``gp``, ``tp`` and ``ps`` margins remains supported.
They sum pointwise tensor products, rather than multiplying separately summed
margins. Numeric, factor and mixed ``re`` margins also support these tensor
constructors with ordinary or matrix inputs. Their default identity precision
or a single supplied precision is supported; multiple precision matrices are
rejected as tensor margins, following upstream. For example,
``te(X, region, bs=['cr', 're'], k=[7, 6], by=L)`` sums paired smooth and random
effect products. Linked matrix-valued tensor ``id`` groups remain unsupported.
The runnable ``examples/example_matrix_functionals.py`` demonstrates a
thin-plate surface and a tensor functional with prediction standard errors.

.. code-block:: python

   import pandas as pd
   from nampy.gam import GAM

   data = pd.DataFrame({
       'y': response,
       'X': list(locations),
       'L': list(quadrature_weights),
   })
   model = GAM(
       formula='y ~ s(X, by=L, bs="tp", k=10)',
       smoothing_method='reml', optimize_smoothing=True,
   ).fit(data=data)

The quadrature weights define the functional predictor; ``sample_weight``
continues to weight observations. Zero quadrature rows contribute no smooth
value. Identifiability depends on whether the sums of the quadrature weights
are constant across observations, following mgcv. Prediction reuses the fitted
basis and accepts new location matrices, including a different number of
quadrature points, or individual scalar locations. Prediction standard errors,
formula offsets and model persistence use the same interfaces as scalar
smooths. Linked matrix-valued ``tp``/``ts`` terms are tested; linked pooling
remains restricted for some other bases.

See :doc:`../notebooks/functional_inputs` for a functional-regression example.

Construction helpers
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   resolve_by_state
   apply_numeric_by
   linear_functional_basis
   build_penalty_definition


Explicit constraint construction
--------------------------------

The construction interface below exposes the raw basis and homogeneous
constraint operator, or absorbs it together with the penalties and prediction
maps. Its supported bases and compositions are described in
:doc:`../user_guide/statistical_gams`.

.. autofunction:: nampy.gam.compiler.construct.smooth_con
