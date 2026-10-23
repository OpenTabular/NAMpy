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
     - Random effects or random coefficients.
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

Construction helpers
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   resolve_by_state
   apply_numeric_by
   linear_functional_basis
   build_penalty_definition

