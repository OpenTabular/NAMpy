Architecture and validation
===========================

Backend organization
--------------------

``nampy.models`` owns the public estimator interfaces. Statistical adapters
compile formulas and delegate fitting to ``nampy.gam``; neural estimators
combine a registered architecture with an objective and delegate training to
``nampy.neural``. The statistical engine uses mgcv and scam as behavioral
references. They are not runtime dependencies or additional NAMpy backends.

.. image:: /_static/backend_architecture.svg
   :alt: Public estimators delegate to statistical or neural backends, which return shared additive prediction contracts
   :class: scientific-figure

The backends share result contracts, including
:class:`nampy.contracts.AdditivePrediction`, rather than numerical family
implementations. A GAM family supplies derivatives and fitting behavior; a
neural distribution supplies target support, output transforms, and loss.
Grouping them by a common response name does not make their internal
parameterizations interchangeable.

Loss-based statistical estimators use :func:`nampy.gam.compile_design` to
obtain an identified basis, fixed penalty, term ownership, response and known
offsets without fitting a likelihood. ``CompiledDesign.evaluate`` evaluates
new rows and their offsets together. The GAM facade and this interface share
formula preprocessing, the compiler, smoothing-override resolution, prediction
matrix construction and smooth-owned offset evaluation. Expectile and quantile
estimators share a neutral loss-estimator base; neither fits a Gaussian model
as a preparation step.

.. _gam-seven-stage-architecture:

The seven stages of the statistical GAM
---------------------------------------

The statistical path in the diagram has seven responsibility stages. These
describe how model intent becomes a numerical fit and then a prediction;
they are not seven interchangeable estimators or seven public method calls.
In particular, the fitting orchestrator has its own internal sequence, and
smoothness selection repeats coefficient solves within stage 5.

.. image:: /_static/gam_seven_stages.svg
   :alt: Seven GAM stages from public interface through formula specification, basis construction, compilation, fitting, fitted results, and prediction; the fitting stage contains repeated coefficient and smoothing updates
   :class: scientific-figure

The running example is ``y ~ s(x, bs='cr', k=8) + z`` with a Gaussian
response. A single input x becomes several basis columns; the linear z term
remains a parametric column. For distributional models, several such predictor
designs are connected by one joint response likelihood.

1. Public interface and fit lifecycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Input:** estimator options, formula or arrays, response family, weights and
known offsets. **Output:** validated fit intent and an isolated fitting session.

``nampy.models`` supplies estimator conventions and delegates to the core
``GAM`` facade in ``nampy/gam/model/api.py``. The facade owns input handling,
formula preprocessing, family setup and public prediction behavior. The core
defaults to fixed smoothing; the estimator adapters request automatic REML
by default where supported. The wrapper choice therefore affects configured
behavior even though both use the same numerical backend.

``FitSession`` in ``model/session.py`` clones the working family and resets
fit-dependent state before fitting. A successful session publishes its state
back to the estimator. This prevents a failed refit from publishing a mixture
of old coefficients and new basis state. Derived caches belong to that fit,
not to a reusable global smooth object.

2. Formula interpretation and canonical specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Input:** model intent and training columns. **Output:** predictor and term
specifications, response/offset information and retained preprocessing state.

``formula/`` parses and extracts the expression; ``specs/`` turns that intent
into canonical predictor specifications. A call such as ``s(x, bs='cr', k=8)``
inside a formula is declarative syntax, not a call to a Python fitting
function. The specification records the basis code, input ownership, k,
by-variable, smoothing identifier and supplied penalty values.

This is where the model decides that there is one smooth of x and one linear
effect of z. It has not yet estimated coefficients or selected lambda.
Training-dependent transformations and categorical encoding retain their
fitted state so new data use the same interpretation. Formula offsets remain
separate known contributions rather than extra estimated coefficients.

3. Basis, penalty and constraint construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Input:** a term specification and its training inputs. **Output:** a fitted
runtime term with basis, penalties, constraints and prediction setup.

``smooths/`` owns runtime term behavior; ``splines/`` implements numerical
basis construction; ``constraints/``, ``penalties/`` and ``coefficients/``
provide the associated transformations. For the cubic example, construction
evaluates spline columns and computes an integrated-curvature penalty.
Centering removes its ambiguity with the intercept. Both the basis and penalty
must be transformed into the same identified coefficient coordinates.

Knots, factor levels, selected kernel construction locations and constraint
maps become fitted term state. They are reused at prediction time. Tensor
products combine marginal constructions; shape-constrained terms additionally
identify coefficient directions requiring positive transforms. These choices
belong here rather than in a generic optimizer. The
:doc:`basis theory <statistical_reference/smooths/theory>` connects these
artifacts to their mathematical meaning.

4. Compilation into a numerical model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Input:** constructed terms and canonical predictor specifications.
**Output:** ``CompiledTerm``, ``CompiledPredictor`` and ``CompiledModel`` state.

``compiler/compile_model.py`` assembles term columns into predictor designs,
records coefficient slices and expands term penalties into compatible blocks.
It preserves which term and predictor own each coefficient and which smoothing
coordinates are shared. A joint location-scale model has several predictor
designs, not a single response column with an extra scale flag.

At this boundary, fitting should receive a numerical model with explicit
ownership and transformations. Compilation does not choose coefficients or
establish convergence. The shared :func:`nampy.gam.compile_design` interface
exposes an identified design, penalty and offset evaluation to fixed-smoothing
loss estimators. Expectile and quantile models reuse this construction without
first fitting a Gaussian likelihood.

5. Coefficient fitting and smoothness selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Input:** compiled designs, response likelihood or loss, weights, offsets and
penalty settings. **Output:** a solved coefficient state with smoothing values
and numerical diagnostics.

``fit/orchestrator.py`` coordinates preparation, selection, solving and
postprocessing. ``fit/solvers/`` and the backend dispatch provide exact
Gaussian, PIRLS, general-family and transformed-coefficient routes where
supported. For ordinary penalized likelihood, the inner target is

.. math::

   -\ell(\theta)+\tfrac12\theta^\mathsf TS_\lambda\theta.

With fixed smoothing, the solver evaluates this problem at configured lambda.
With automatic smoothing, ``fit/selection/`` evaluates a criterion such as
REML or GCV and updates the free log smoothing parameters. Each candidate
requires another coefficient solve. Analytic derivatives account for how the
coefficient optimum changes with smoothing. EFS supplies a different update
strategy for its supported routes; not every family supports every optimizer.

For SCOP terms, the map from free parameters to spline coefficients is
nonlinear. The likelihood and penalty derivatives must follow that map.
Consequently the constrained route is a fitting branch within this stage,
not a cosmetic constraint applied after an ordinary GAM is fitted. Loss-based
estimators instead provide their own solver and fixed-penalty objective over
the shared compiled design. See :doc:`statistical_reference/selection/theory`
for the inner/outer distinction.

6. Fitted result and inference artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Input:** the exact solved state and its compiled model. **Output:** canonical
fitted artifacts, covariance/influence quantities where available, and summaries.

``results/`` defines ``GAMResult`` and ``GAMFitResult``; ``fit/state.py``
defines ``FitCoreSolution``. ``GAMResult`` is assembled progressively as
compilation, solving and summary construction finish. Its outer aggregate
is replaced at those transitions; internal compiled terms retain their fitted
construction state. Replacing a solution invalidates its derived summary.

Coefficients, covariance, EDF and diagnostics must describe the same fitted
design and smoothing endpoint. They cannot be recomputed from a superficially
similar new basis. Conditional covariance and smoothing-uncertainty corrections
are different artifacts and depend on fitting-route support. Finite arrays
are not a convergence certificate: solver and outer stopping information
remain part of assessing a fit.

7. Prediction, inference and diagnosis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Input:** new rows together with retained preprocessing, runtime terms and
fitted coefficients. **Output:** predictions, parameter-specific contributions,
uncertainty summaries and diagnostic objects.

``predict/`` evaluates the already-fitted term setup and assembles new-data
designs. On the link scale, the accounting is

.. math::

   \eta_{\rm new}=B_{\rm new}\widehat\theta+o_{\rm new}.

The family then maps predictors to response outputs or natural parameters.
Prediction does not reconstruct knots from the new rows or rerun smoothing
selection. Formula offsets are evaluated under the retained formula contract;
a separate fit-time offset is not automatically reused for new observations.

``inference/`` propagates supported coefficient covariance through prediction
or functional matrices, while ``diagnostics/`` evaluates residuals, basis
adequacy, concurvity and other checks. ``AdditivePrediction`` records predictor
contributions so their reconstruction can be verified before presentation.
Distributional contributions are additive on their own predictor scales;
nonlinear parameter transforms do not generally preserve that sum on the
response scale. The shared explanation and plotting layers consume these
contracts without taking over fitting.

Where a new feature belongs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A new basis mainly extends stage 3 and declares its capabilities for stages
4, 5 and 7. A new likelihood supplies family domains, links and derivatives
consumed by fitting and prediction. A new selection criterion extends stage 5,
while a new diagnostic consumes stage 6 state. Preserving these boundaries
lets models reuse formula handling, prediction coordinates and output contracts
instead of copying the complete lifecycle. The specific extension points follow.

Extension points
----------------

* **Architecture:** add a forward model and configuration under
  ``nampy/neural/architectures/`` and ``nampy/neural/configs/``, then declare
  capabilities through :class:`nampy.neural.registry.NeuralArchitecture`.
  :func:`nampy.models.estimator_family` generates compatible estimator classes.
  The :doc:`custom-model guide <user_guide/custom_models>` explains feature
  blocks, output width, and named contribution tensors.
* **Neural objective or distribution:** put target encoding, per-row loss,
  prediction semantics, and metrics in ``objectives.py`` or
  ``neural/distributions/``. Keep them independent of the forward architecture.
* **Smooth or statistical fitting change:** work in ``nampy/gam/`` and identify
  the relevant basis, penalty, coefficient solver, or smoothing-selection
  layer. Preserve explicit compatibility checks and the upstream statistical
  objective.
* **New statistical loss:** reuse ``compile_design`` and the shared estimator
  lifecycle, supplying the coefficient solver and per-row scoring loss.
  Target shape validation, term contributions, prediction offsets and plotting
  stay in the shared layer. Statistical exports are declared in
  ``nampy/models/_statistical_registry.py``; general-family constructors and
  supported options are declared in ``nampy/gam/families/registry.py``.
  Documentation discovers membership from these inventories and requires an
  authored example for every exposed extension.
* **Execution contract:** the facade owns splitting, train-only preprocessing,
  seeded construction, offsets, persistence, and inference batching. Fixed
  linear designs and native-training providers expose specialized solves
  without moving those shared responsibilities into architecture classes.
  See :doc:`api/contracts` and :doc:`api/neural_training`.

Validation approach
-------------------

Interface tests check shapes, domains, cloning, persistence, and reconstruction
of the reported link prediction. Reconstruction verifies component accounting;
it does not establish uniqueness, effect stability, or calibration. Metadata
must distinguish parameter identity from feature dependence, and global terms
must not be presented as marginal feature curves. These invariants are checked
in ``tests/neural/test_documented_component_contracts.py`` and
``tests/docs/test_example_semantics.py``.

Numerical comparisons use versioned fixtures for design matrices, penalties,
fitted quantities, covariance, and diagnostics. Each case selects tolerances
appropriate to its representation and numerical objective. Agreement in one
case is evidence for that case, not the entire family/optimizer Cartesian
product. Finite coefficients are also distinct from convergence: the
:doc:`optimizer route report <development/optimizer_matrix>` currently checks
the former and expected rejection. Convergence must be assessed separately from finite output.

Executed examples test the public workflow and preserve useful outputs.
Notebook records identify source content, package versions, and code hashes;
the structural checker verifies links, anchors, navigation, and saved execution
records. These checks complement numerical tests: a runnable example can
still contain incorrect logic or misleading interpretation. Build and execution
commands have one home in :doc:`contributing`.

Detailed evidence
-----------------

* :doc:`development/notebooks/02_statistical_gams` exercises statistical
  interfaces; focused comparisons live in `statistical parity tests
  <https://github.com/OpenTabular/NAMpy/tree/main/tests/parity>`_, smooth,
  optimization, diagnostics, and SCAM test directories.
* :doc:`development/notebooks/04_distributional_models` is the consolidated
  neural task/objective report. :doc:`development/notebooks/03_neural_models`
  retains architecture-specific checks. Their small fitting budgets are
  interface evidence, not comparative performance benchmarks.
* :doc:`development/optimizer_matrix` contains the generated route sample.
  User-facing restrictions belong in :doc:`compatibility`.
* The `fixture manifest <https://github.com/OpenTabular/NAMpy/blob/main/tests/reference_fixtures/manifest.json>`_
  records upstream versions, namespaces, and the declared mgcv source
  difference. Check it when interpreting numerical comparisons.

.. _specialized-neural-evidence:

Specialized capability checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Higher-order SIAN terms are compared with upstream states in
`test_sian_reference.py <https://github.com/OpenTabular/NAMpy/blob/main/tests/neural/test_sian_reference.py>`_.
Sparse NBM tensors and SPAM local expansions are checked in
`test_nbm_spam_reference.py <https://github.com/OpenTabular/NAMpy/blob/main/tests/neural/test_nbm_spam_reference.py>`_.
Public estimator local-importance calls are exercised in
`test_neural_estimator_fit_smoke.py <https://github.com/OpenTabular/NAMpy/blob/main/tests/neural/test_neural_estimator_fit_smoke.py>`_,
and registry declarations in
`test_neural_registry_objectives.py <https://github.com/OpenTabular/NAMpy/blob/main/tests/neural/test_neural_registry_objectives.py>`_.
These focused tests provide evidence beyond the consolidated report's small
main/task sweeps; they do not imply exhaustive coverage of every interaction
order, sparsity configuration, or objective combination.

Statistical extension scope
---------------------------

Implemented extensions cover fixed-smoothing expectiles and quantiles, joint
location/scale families, simultaneous linear-functional inference, piecewise
exponential survival data and exposure-history designs. Supported inputs and
limitations are defined in their reference pages. Numerical comparison details
and upstream mappings are maintained in :doc:`development/neighbor_parity`.

.. toctree::
   :maxdepth: 1

   Statistical extension validation <development/neighbor_parity>
