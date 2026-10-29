.. _statistical-gam-tools:

Statistical GAMs in Python
==========================

This workflow fits a mean model, checks its smoothness, predicts held-out rows,
and inspects fitted effects. The :doc:`statistical reference </statistical_reference/index>`
contains complete argument contracts, supported combinations and fitting limits.

The :doc:`GAM model guide </models/gam>` develops the theory behind this
workflow: response families and links, basis expansions, penalized fitting,
smoothness selection and uncertainty. Use the worked steps here to connect
those concepts to a fitted model.

1. Choose an interface
----------------------

Use ``GAMRegressor`` for a scalar regression response and a scikit-learn
interface. ``GAMClassifier`` models binary labels; ``GAMLSS`` models multiple
distribution parameters. The core ``GAM`` exposes the full statistical engine.
Compare their :doc:`estimator contracts </statistical_reference/estimators/index>`.

Here is reproducible data for the complete workflow:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from nampy.models import GAMRegressor

   rng = np.random.default_rng(7)
   n = 160
   data = pd.DataFrame({
       "x": rng.uniform(0, 1, n),
       "z": rng.normal(size=n),
       "group": pd.Categorical(np.resize(["control", "treatment"], n)),
   })
   data["y"] = (np.sin(5 * data.x) + 0.4 * data.z
                + 0.3 * (data.group == "treatment") + rng.normal(0, 0.2, n))
   train = data.sample(frac=0.75, random_state=7)
   test = data.drop(train.index)

.. _gam-formula-language:

2. Specify a formula
--------------------

The formula expresses which effects you want to estimate. A smooth represents
a nonlinear effect, while ordinary columns contribute parametric terms:

.. code-block:: python

   formula = "y ~ s(x, bs='cr', k=8) + z + group"

This model has a curved effect of ``x``, a linear effect of ``z`` and a group
contrast. A smooth is centered against the intercept under its basis-specific
identifiability rules. Interpret contributions on the additive predictor scale.

Use ``te(x,z)`` for a full joint surface, or separate main smooths and
``ti(x,z)`` for an interaction decomposition. The
:doc:`formula visual reference <formula_visual_reference>` shows the resulting
curves and surfaces. The :doc:`formula API </api/gam_formula>` documents syntax,
transformations, grouped responses and offsets; the
:doc:`smooth inventory </statistical_reference/smooths/index>` compares bases.

Array-valued DataFrame cells can represent evaluation locations for functional
predictors. Their pairing, quadrature weights and supported compositions are
defined in :ref:`matrix-functional-smooths`. Explicit parametric penalties and
homogeneous smooth constraints are documented in
:doc:`/statistical_reference/penalties`.

3. Choose a response family
---------------------------

A family describes response support, mean/variance behavior and the link
between the additive predictor and the response. Gaussian identity is suitable
for this real-valued example. Counts, proportions and positive measurements
need the corresponding response models; start with the
:doc:`family inventory </statistical_reference/families/index>`.

For changing spread or shape, use :doc:`distributional regression <gamlss>`.
Its parameter predictors have separate formulas and can have different natural
and low-level response outputs. Each family page defines those column meanings,
links, target encoding and supported adapters.

.. _smoothness-and-basis-dimension:

4. Fit and assess smoothness
----------------------------

The basis dimension ``k`` limits representational capacity. The penalty chooses
how much of it the fitted curve uses. A larger ``k`` permits flexibility;
it does not instruct the model to fit a rougher curve. A smooth has the form

.. math::

   f_j(x)=\sum_{r=1}^{K_j}b_{jr}(x)\theta_{jr},

and coefficients balance fit against roughness through

.. math::

   -\ell(\theta)+\frac12\sum_m\lambda_m\theta^\mathsf T S_m\theta.

Effective degrees of freedom (EDF) summarize the flexibility left after
penalization and need not be integers. The adapter selects REML smoothness by
default for this supported Gaussian model:

.. code-block:: python

   model = GAMRegressor(formula=formula, family="gaussian").fit(train)
   print(model.gam_.summary())
   print(model.gam_.k_check(n_rep=50, seed=7))

Inspect the fit's EDF and residual patterns, and use ``k_check`` to decide
whether to investigate a larger basis. It is a diagnostic, not an acceptance
test. The :doc:`first GAM tutorial </tutorials/first_gam>` illustrates these
choices with fitted plots. Criteria, optimizers and tolerances have separate
roles; their contracts are in :doc:`/statistical_reference/selection/index`
and :doc:`/statistical_reference/controls/index`.

5. Predict held-out observations
--------------------------------

.. code-block:: python

   prediction = model.predict(test)
   standard_error = model.standard_errors(test)
   comparison = pd.DataFrame({
       "observed": test.y,
       "prediction": prediction,
       "mean_se": standard_error,
   })
   print(comparison.head())
   print("Held-out R2:", model.score(test, test.y))

The predictions and standard errors each have shape ``(n_test,)`` here.
The SE describes uncertainty about the fitted mean, not the spread of a future
observation. Under nonlinear links, link and response predictions differ.
:doc:`Prediction choices </howto/prediction_outputs>` and the
:doc:`predict contract </statistical_reference/methods/predict>` define those
outputs. New rows must supply fitted source features and formula-offset columns.
Separate fit-only offsets must be supplied again when predicting; see
:doc:`/howto/offsets`.

6. Diagnose and interpret
-------------------------

.. code-block:: python

   term_figures = model.plot(residuals=True, pages=1)
   check = model.gam_.gam_check(k_rep=50, seed=7)
   components = model.predict_components(test)
   components.validate_additive_reconstruction()

The intercept, terms and offset reconstruct the link predictor. For this
identity-link model the response scale is the same; this does not hold for all
families. Examine term shapes alongside residual structure and the domain in
which you observed each predictor. Correlated predictors can make individual
effects unstable even when total predictions are stable.

A term plot shows one contribution. A conditional full-model surface instead
varies selected features while fixing the others:

.. code-block:: python

   surface = model.vis(
       view=["x", "z"], cond={"group": "control"},
       type="response", plot_type="contour", n_grid=40, too_far=0.1,
   )
   surface_figures = surface["figures"]

For slopes, choose a basis with derivative support, such as ``ps`` or ``bs``.
Region maps need polygon geometry; spherical displays use latitude/longitude.
The :doc:`plotting reference </api/plotting>` retains worked surface, derivative,
region, hemisphere and QQ examples, with their complete rendering options.

Use :doc:`/statistical_reference/inference` to interpret covariance, pointwise
versus simultaneous intervals, EDF and information criteria together. The
:doc:`diagnostic method index </statistical_reference/methods/index>` documents
summary, ANOVA, residuals, concurvity and basis checks. Numerical restrictions
and validation conventions belong to their method and family pages and
:doc:`/compatibility`.

Fitted distributed-lag workflows
--------------------------------

``nampy.gam.structured.DistributedLagBasis`` fits exposure and lag marginal
bases through the existing formula compiler. Histories have a fixed width and
order: lag zero, lag one, through ``max_lag``. ``exposure_history`` constructs
these rows without crossing contiguous group boundaries. ``complete_rows``
identifies missing histories; missing lags are never replaced by zeros.

Call ``fit`` on training histories and reuse ``transform`` for new histories.
``save_model``/``load_model`` preserve the fitted marginal knots, identification
maps and layout; transformation never refits on query rows. The exposure
intercept is excluded and the lag intercept is retained. ``penalty`` returns
the fixed tensor penalty in transformed column order. This transformer does
not fit response coefficients or select smoothing parameters.

``cumulative_contrast(exposure, reference)`` compares constant exposure paths
across every lag on the link scale. Optional explicit ``coefficient_indices``
and ``n_coefficients`` embed that matrix in full model coordinates. Pass it to
``linear_functional_interval`` for pointwise or finite-grid simultaneous
statistical intervals, retaining cross-lag covariance. Response-scale nonlinear
contrasts need paired transformed draws; this does not supply neural coefficient
uncertainty or establish a causal interpretation.
