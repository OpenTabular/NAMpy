Parametric penalties and explicit smooth constraints
====================================================

``para_pen`` is the Python spelling of mgcv's ``paraPen``. It maps original
parametric formula labels to one or more matrices, in that term's fitted column
order. For example, this shrinks the coefficient of ``x`` while fitting a smooth
of ``z``:

.. code-block:: python

   model = GAM(
       formula="y ~ x + s(z, bs='cr', k=7)",
       para_pen={"x": {"penalties": [np.eye(1)], "sp": [-1]}},
       optimize_smoothing=True,
       smoothing_method="reml",
   ).fit(data=df)

Penalty entries
---------------

.. list-table::
   :header-rows: 1

   * - Operation
     - Result
   * - ``penalties``
     - One or more matrices in the fitted term's column order. Factor terms use expanded contrast columns and matrix-valued terms use their expanded columns. NAMpy uses the matrices at the supplied scale.
   * - ``rank``
     - Optional rank, one integer per matrix.
   * - ``sp``
     - Nonnegative values fix parameters. Negative values or an omitted ``sp`` request estimation when smoothing optimization is enabled.
   * - ``L``
     - Optional map from log smoothing parameters to log penalty multipliers. The matrix may be rectangular with finite signed or fractional entries.

Parametric smoothing parameters precede smooth parameters, in formula-term
order. As in mgcv, unmatched labels are ignored. The interface supports
single-formula fits and excludes the intercept as a penalty target.

Log-scale maps and selection
----------------------------

For positive underlying parameters, the full multipliers are
``np.exp(L @ np.log(sp))``. Fixed parameters contribute through the fixed
log-SP offset. A zero row gives a multiplier of one. Fixed zeros use mgcv's
effective-zero construction. See ``examples/example_parametric_penalty_map.py``
for two underlying parameters mapped to three penalties.

When combining nonidentity parametric maps with smooth penalties, setup follows
mgcv's combined ``L`` row/column ordering exactly. In mgcv 1.9-4 this ordering
interleaves setup coordinates differently from a simple parametric block followed
by a smooth block. Fixed and selected fits follow the captured upstream state.

* **Optimizers.** Newton and supported alternative outer optimizers use mapped
  analytic derivatives. Automatic EFS is guarded because its upstream branch
  omits ``L`` and its offset.
* **Initialization.** Free map columns must be used. Automatic initialization
  requires a full-rank free map. Explicit initial values can bypass that step.
* **Bounds and transforms.** Nonidentity maps with positive ``min_sp`` require
  a separate full-penalty ``H`` implementation. Transformed-coefficient fits
  also exclude mapped penalties.

Variance components
-------------------

``gam_vcomp(rescale=False)`` separates underlying parameter names from full-penalty
standard deviations and uses the accepted Gaussian REML variance. Rescaling
requires the corresponding parametric scale metadata.

For ordinary partially pooled group intercepts, ``GAM.gam_vcomp()`` reports
component standard deviations on the additive predictor scale; random-slope
units also reflect the covariate scaling. Available confidence limits depend
on the fitted likelihood and outer curvature, and should be read with the
returned rank and finite-bound information. A fixed-penalty fit with no free
smoothing parameters returns ``None``. The :doc:`method reference
<methods/gam_vcomp>` specifies the output fields; use the
:doc:`group-smooth tutorial </notebooks/group_smooths>` to connect these
components to partial pooling.

Explicit smooth constraints
---------------------------

``constraints={1: C}`` replaces the first smooth's centering with
``C @ beta_raw = 0``. A unique smooth label can stand in for the one-based index.

* **Supported fits.** Use ``cr``, ``cc``, ``ps``, ``cp`` or ``bs`` without ``by``,
  point constraints or linked IDs.
* **Constraint matrix.** ``C`` acts on the uncentered basis. Use finite,
  independent rows and leave at least one coefficient. A zero-row matrix
  removes the local constraint while retaining global identifiability rules.
  Columns with cancelling nonzero row sums are outside upstream's localized
  absorption contract.
* **Shrinkage cubic splines.** Raw ``smooth_con`` inspection supports invariant
  checks for ``cs``. Its unequal penalties on arbitrary null eigenvectors make
  explicit constrained fits sensitive to LAPACK orientation. Use ``cr`` with
  ``select=True`` for a portable penalized null space.

Multiple B-spline penalties
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Derivative-penalized B-splines (``bs='bs'``) also support multiple derivative
penalties through ``m``. The same constraint transform applies to every
penalty, and ``select=True`` adds a null-space penalty after constraint
absorption when a null space remains. See ``examples/example_bs_constraints.py``
for a runnable fit with an explicit coefficient-sum constraint.

For an empty constraint matrix combined with multiple B-spline penalties and
``select=True``, default GCV performance iteration can stop at different
endpoints in NAMpy and mgcv. Fixed-SP reconstruction agrees; the targeted
automatic-selection comparison uses the same tighter ``mgcv_tol=1e-12`` in
both implementations. This is an optimizer stopping limitation, not a
constraint-operator difference.

Inspect the raw basis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nampy.gam.compiler.construct import smooth_con
   from nampy.gam.specs import TermSpec, build_smooth_spec

   term = TermSpec(
       kind="smooth", features=("x",), label="s(x)",
       smooth_spec=build_smooth_spec(special="s", bs="cr", k=7),
   )
   raw = smooth_con(term, df[["x"]].to_numpy(), ["x"], absorb_cons=False)
   C = raw.fit_constraint_operator
   constrained = smooth_con(term, df[["x"]].to_numpy(), ["x"], constraints=C)
   X_new = constrained.compiled_term.predict_matrix(new_df[["x"]].to_numpy())

The returned object contains ``compiled_term``, ``penalty_specs`` and
``fit_constraint_operator``. Absorption transports the basis, every penalty,
prediction and derivative coefficient maps together. ``absorb_cons=False`` is
a construction/inspection control; fitted GAMs always absorb constraints,
matching mgcv's high-level fitting route. It is not a switch to fit an
unconstrained model accidentally.

See :doc:`/statistical_reference/GAM` for the types and defaults of ``para_pen`` and ``constraints`` and :doc:`/extension_reference/index` for runtime construction APIs.

Complete example
----------------

.. literalinclude:: /_examples/reference/statistical_reference/penalties.py
   :language: python

:download:`Download the complete penalty and constraint example </_examples/reference/statistical_reference/penalties.py>`.
