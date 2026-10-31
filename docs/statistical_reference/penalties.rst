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

Each entry accepts ``penalties``, optional ``rank`` (one integer per matrix),
``sp`` (nonnegative for fixed, negative for estimated), and ``L``. Omitting
``sp`` marks its parameters for estimation; smoothing optimization must be
enabled to estimate them. Matrices are not rescaled. A factor term uses its
expanded contrast columns; a matrix-valued term uses its expanded columns.
Several penalties may act on the same term. ``L`` maps log underlying smoothing
parameters onto log penalty multipliers. It may be rectangular and contain
finite signed or fractional entries: with positive underlying parameters,
the full multipliers are ``np.exp(L @ np.log(sp))``. Fixed parameters are
removed from optimization and contribute through the fixed log-SP offset.
Zero rows specify a constant multiplier of one. Fixed zeros use mgcv's
effective-zero construction. See ``examples/example_parametric_penalty_map.py``
for a fit with two underlying parameters and three penalties.
Parametric smoothing parameters precede smooth parameters, in formula-term
order. As in mgcv, unmatched labels are ignored and the intercept is not a
parametric-penalty target. Multi-formula ``para_pen`` is rejected, including
general-family formula lists with one predictor.

When combining nonidentity parametric maps with smooth penalties, setup follows
mgcv's combined ``L`` row/column ordering exactly. In mgcv 1.9-4 this ordering
differs from simply prepending a parametric block to the smooth map; fixed and
selected fits follow the captured upstream setup state. Automatic EFS selection
with nonidentity maps is guarded: upstream's EFS branch omits ``L`` and its
offset, causing rectangular-map failures or inconsistent square-map states.
Newton and supported alternative outer optimizers use the mapped analytic
derivatives. Unused free map columns are rejected rather than estimated.
More generally, automatic initialization rejects rank-deficient free maps,
matching source initialization; explicit initial smoothing parameters may
bypass that initialization. Positive ``min_sp`` with nonidentity maps requires
a separate full-penalty ``H`` implementation and remains guarded. Mapped
penalties are unavailable for transformed-coefficient fits.
``gam_vcomp(rescale=False)`` reports underlying parameter names separately
from full-penalty standard deviations and uses the accepted REML variance for
Gaussian fits. Rescaling parametric variance components remains guarded where
the required scale metadata is unavailable.

For ordinary partially pooled group intercepts, ``GAM.gam_vcomp()`` reports
component standard deviations on the additive predictor scale; random-slope
units also reflect the covariate scaling. Available confidence limits depend
on the fitted likelihood and outer curvature, and should be read with the
returned rank and finite-bound information. A fixed-penalty fit with no free
smoothing parameters returns ``None``. The :doc:`method reference
<methods/gam_vcomp>` specifies the output fields; use the
:doc:`group-smooth tutorial </notebooks/group_smooths>` to connect these
components to partial pooling.

``constraints={1: C}`` replaces the first smooth's default centering with
``C @ beta_raw = 0``. A unique smooth label may replace the one-based index.
Fitting supports ``cr``, ``cc``, ``ps``, ``cp`` and ``bs`` without ``by``,
point constraints, or linked IDs. ``C`` acts on the uncentered basis, must be
finite with independent rows, and must leave at least one coefficient.
A matrix with zero rows requests no local constraint; global model
identifiability conditions still apply. Columns with cancelling nonzero row
sums are rejected because upstream's localized absorption cannot represent them.
Other bases/compositions remain guarded rather than accepting a constraint in
a different raw parameterization.
Explicit ``cs`` constraints are guarded: mgcv assigns unequal shrinkage to
numerically arbitrary null eigenvectors, which changes fitted predictions
across LAPACK implementations. Use ``cr`` with ``select=True`` for a portable
penalized null space. Raw ``smooth_con`` inspection of ``cs`` remains available
with invariant penalty and constraint checks.

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

Inspect the raw basis before specifying ``C``:

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
