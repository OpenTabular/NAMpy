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

See :ref:`howto-user-guide-shape-constrained-gams`.

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

Basis and weight conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same weighted or unit-weight summation is supported by ``cr``,
``cc``, ``ps``, ``cp``, ``bs``, ``ds``, ``gp`` and ``sos``. Duchon and GP
terms accept paired multivariate locations; spherical terms use paired
latitude and longitude matrices.

* **Penalty and centering.** Penalties are scaled on the pointwise basis before
  summation. Constant row sums retain the upstream centering constraint even
  when individual weights vary.
* **Repeated locations.** The setup pools coordinates in column-major order.
  When fewer than 90% of point tuples are unique, it packs them to their first
  occurrences. Prediction expands the pointwise basis before summation.
* **Shrinkage cubics.** Matrix ``cs`` fits, including distinct locations, are
  guarded because unequal null-space shrinkage depends on eigenvector
  orientation. Scalar ``cs`` remains supported.

Adaptive and bounded surfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Operation
     - Result
   * - Adaptive ``ad``
     - Supports weighted or unit-weight 1D matrices and unit-weight paired 2D matrices. Weighted 2D prediction loses the ``by`` variable in mgcv 1.9-4 and is guarded.
   * - Soap ``so`` / ``sf`` / ``sw``
     - Supports paired location matrices and weighted sums. For known boundaries, use ``sw``. Upstream ``smoothCon`` cannot aggregate the boundary offsets of ``so`` or ``sf``.

Random effects and grouped smooths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Random effects and MRF.** ``re`` and full- or reduced-rank ``mrf`` accept
  array-valued categorical cells. Factor levels are pooled across cells, and
  labels such as ``"01"`` and ``"1"`` stay distinct. MRF requires an adjacency
  or penalty specification and rejects unknown regions. Random effects give
  unseen levels zero contribution.
* **Numeric-by-factor random effects.** ``s(region, by=L, bs='re')`` accepts
  lists of region labels. ``s(X, region, by=L, bs='re')`` multiplies the
  numeric and categorical effects at each paired location before summing.
* **Factor smooths.** For ``fs``, use ``xt={'bs': 'cr'}`` or ``xt={'bs': 'ps'}``
  with matrix locations and a scalar factor per observation. Both bases support
  varying, unit and constant-row-sum weights, including repeated locations.
  For ``tp``, use one metric coordinate and ``m=1``, as in
  ``s(X, group, bs='fs', k=6, m=1, by=L)``. Matrix ``fs`` rejects the default
  ``tp`` because parity is unresolved for its independent null penalties.
* **Sum-to-zero factor smooths.** For ``sz``, matrix functionals support ``tp``
  with one or two metric coordinates, ``cr`` and ``ps``, with one or more
  scalar factors per observation. Unit, varying and constant-row-sum weights
  follow upstream constraints. Repeated ``cr`` and ``ps`` locations are also
  validated. Both ``fs`` and ``sz`` pass ``m`` to their ``tp`` bases.
* **Soap bases.** With paired matrix coordinates and scalar factors, use
  ``fs(sf)``, ``sz(sf/sw)`` or ``sz(so)`` with a fully known boundary. See the
  source offset conventions above.

Matrix-valued factors
~~~~~~~~~~~~~~~~~~~~~

To supply a matrix-valued factor, store each row as a ``pandas.Categorical``
with the same declared categories and ordered flag.

.. code-block:: python

   df["group"] = pd.Series(
       [pd.Categorical(row, categories=["a", "b", "c"]) for row in groups],
       dtype=object,
   )

``groups`` has the same row count and quadrature width as the metric matrices.
Each location may have its own factor level. The categorical rows carry the
equivalent of R's factor-plus-dim metadata; plain arrays of labels do not.

* **Factor smooths.** For ``fs``, use single-penalty ``sf``, one-dimensional ``tp`` with
  ``m=1``, or ``ps`` with ``m=[2,1]``. Multiple independent null penalties
  require parity of the penalty operator as well as the function space.
* **Sum-to-zero factor smooths.** For ``sz``, use ``cr``, ``ps``, ``tp`` or soap bases.
* **Multiple factors.** Every joint training group must be observed. Empty
  joint groups and multi-null ``fs`` are guarded by NAMpy's parity checks.
* **Width.** Use at least two columns because upstream converts a one-column
  factor matrix to a numeric vector. At prediction, width may change if
  factor levels and pairing with metric locations and weights are preserved.

See ``examples/example_matrix_factor.py`` for a complete soap fit.

Tensor-product functionals
~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor functionals sum pointwise tensor products across paired locations.

* **Tested margins.** ``te``, ``ti`` and ``t2`` support ``cr``, ``ps``, ``tp``,
  ``ds``, ``gp``, ``sf`` and ``sw``. ``te`` and ``ti`` also support ``bs``.
* **B-spline tensors.** Matrix inputs to ``t2`` support ``bs`` when each null space has at
  most one dimension (for example ``m=[[3, 1], [3, 1]]``), or when only the
  highest range-space order is retained (``ord=[2]`` for two margins). Mixed
  range/null blocks with multidimensional ``bs`` null spaces remain guarded
  pending portable normalization. Ordinary scalar-input ``t2`` is supported.
* **Random-effect margins.** Numeric, factor and mixed ``re`` support ordinary
  or matrix inputs with identity or a single supplied precision matrix.
  Upstream excludes multiple precisions as tensor margins.
  ``te(X, region, bs=['cr', 're'], k=[7, 6], by=L)`` sums paired smooth and
  random-effect products.
* **Linked IDs.** Linked matrix-valued tensor ``id`` groups are unsupported.

The runnable ``examples/example_matrix_functionals.py`` demonstrates a
thin-plate surface and a tensor functional with prediction standard errors.

Fitting and prediction
~~~~~~~~~~~~~~~~~~~~~~

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

* **Weights.** Quadrature weights define the functional predictor and
  ``sample_weight`` weights observations. Zero quadrature rows contribute zero.
  Identifiability follows mgcv's constant-row-sum rule.
* **New locations.** Prediction reuses the fitted basis and accepts matrices
  with a different number of quadrature points or individual scalar locations.
* **Shared interfaces.** Prediction standard errors, offsets and persistence
  use the scalar-smooth interfaces. Linked matrix ``tp`` and ``ts`` are tested.
  Pooling remains restricted for some other bases.

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
