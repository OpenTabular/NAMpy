:orphan:

Smooth Terms
============

The :doc:`grouped smooth reference </statistical_reference/smooths/index>`
contains formula arguments, basis distinctions and runnable examples.
For constructors and basis internals, use :doc:`/extension_reference/smooths/index`.

.. _gam-smooth-bases:

Basis lookup
------------

See :doc:`/statistical_reference/smooths/index`.

.. _soap-film-smooths:

Bounded spatial domains
-----------------------

See :doc:`/statistical_reference/smooths/so`.

.. _structured-smooth-bases:

Group effects
-------------

See :doc:`/statistical_reference/smooths/index`.

.. _shape-constrained-basis-codes:

Shape constraints
-----------------

See :doc:`/user_guide/shape_constrained_gams`.

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
multiple independently penalized null directions have unresolved parity.
Univariate ``tp`` with ``m=1`` has a single null direction and is supported,
for example ``s(X, group, bs='fs', k=6, m=1, by=L)``. The requested ``m``
is passed to thin-plate bases inside both ``fs`` and ``sz``.
Alternatively specify an accepted base explicitly, for example
``s(X, group, bs='fs', k=7, xt={'bs': 'cr'}, by=L)``.

``sz`` matrix functionals support ``tp`` (one or two metric coordinates),
``cr`` and ``ps`` bases, with one or more scalar factor columns per observation.
Unit, varying and constant-row-sum weights follow upstream constraint handling;
repeated-location ``cr``/``ps`` cases are also validated. Paired matrix soap
coordinates with scalar factors support ``fs(sf)``, ``sz(sf/sw)`` and fully
known-boundary ``sz(so)``; their source offset conventions are described above.

To supply a matrix-valued factor, store each row as a ``pandas.Categorical``
with the same declared categories and ordered flag. For example:

.. code-block:: python

   df["group"] = pd.Series(
       [pd.Categorical(row, categories=["a", "b", "c"]) for row in groups],
       dtype=object,
   )

``groups`` has the same row count and quadrature width as the metric matrices.
Each location may have its own factor level. The categorical rows carry the
equivalent of R's factor-plus-dim metadata; plain arrays of labels do not.
With matrix-valued factors, ``fs`` supports single-penalty ``sf``, ``tp`` with
``m=1`` in one dimension, and ``ps`` with ``m=[2,1]``. Multiple independent null
penalties remain guarded: matching the function space alone does not preserve
the independently weighted null-space penalty operator across representations.
``sz`` supports the documented ``cr/ps/tp`` and soap bases.
With multiple matrix-valued factors, every joint training group must be
observed; empty joint groups remain guarded pending matching rank and side
conditions. Both this restriction and the multi-null ``fs`` restriction are
NAMpy parity limitations, rather than upstream input rejections.
Use at least two columns: upstream converts a one-column factor matrix to a
numeric vector. Prediction may change the quadrature width while preserving
the factor levels and pairing with the metric coordinates and weights.
See ``examples/example_matrix_factor.py`` for a complete soap fit.

``te``, ``ti`` and ``t2`` support paired matrix coordinates with tested
``cr``, ``ps``, ``tp``, ``ds``, ``gp``, ``sf`` and ``sw`` margins.
``te`` and ``ti`` additionally support ``bs`` margins.
Matrix ``t2`` also supports ``bs`` margins when each B-spline null space has
at most one dimension, such as ``m=[[3, 1], [3, 1]]``, or when only the
highest range-space order is retained, such as ``ord=[2]`` for two margins.
These cases avoid the ambiguous null-space metric. Mixed range/null terms
with multidimensional ``bs`` null spaces remain rejected because portable
null-space normalization parity is unresolved. ``te``/``ti`` with ``bs``
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

.. currentmodule:: nampy.gam.smooths

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
:doc:`/statistical_reference/penalties`.

.. autofunction:: nampy.gam.compiler.construct.smooth_con
