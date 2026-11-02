Architecture and validation
===========================

Backend organization
--------------------

``nampy.models`` owns the public estimator interfaces. Statistical adapters
compile formulas and delegate fitting to ``nampy.gam``; neural estimators
combine a registered architecture with an objective and delegate training to
``nampy.neural``. The statistical engine uses mgcv and scam as behavioral
references. They are not runtime dependencies or additional NAMpy backends.

Internal functions and state fields describe their mathematical responsibilities,
rather than upstream solver numbers. For example, canonical Gaussian coefficient
solving and penalty-coordinate transforms have explicit names. Upstream symbols
such as ``gam.fit3`` remain in provenance comments and validation fixtures so the
reference algorithms can still be traced. Conventional local matrix notation
such as ``X``, ``Q`` and ``R`` is retained where it clarifies the algebra.

The public task packages are ``distributions`` (probability queries),
``diagnostics`` (residuals, centiles, effects, importance and Fisher information),
``plotting`` (rendering), ``calibration`` and ``survival``. Shared distribution
contracts live with probability laws; generic feature/additive contracts remain
in ``nampy.contracts``. See :ref:`howto-user-guide-api-and-families` for canonical
imports and the generated backend support table.

``models/comparison.py`` owns candidate fitting and held-out ranking.
Probability laws have no fitting or diagnostics dependency; residuals consume
probability predictions through ``nampy.diagnostics``. Neutral table adaptation
and offset validation live in ``_data.py`` and ``_offsets.py``. Fitted feature-schema
preparation also lives in ``_data.py``; backend training imports this neutral owner.
Statistical prediction covariance calculations remain in ``gam/inference``.

``model.capabilities()`` composes existing family and architecture declarations
with fitted-state validators. It reports supported, conditional or unsupported
operations without evaluating predictions or differentiating the fitted model.
See :ref:`howto-user-guide-model-workflow` for the shared task contracts.
``model.capabilities()`` reports the declarations for the configured model.

Training families are named by their owner: ``gam/families`` supplies statistical
likelihood derivatives and ``neural/families`` supplies Torch losses. Their
registries own constructor options and probability-prediction adapters.
Statistical row-correlation transforms live in ``gam/correlation``. Quantile loss
and the common parameterized-response objective base live in ``neural/objectives``.

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

The statistical path has seven responsibility stages. Select a stage to
see the function that owns it and the variables that function writes. A
trailing underscore is state written by a fit. Smoothness selection repeats
coefficient solves inside stage 5.

.. image:: /_static/gam_seven_stages.svg
   :alt: Seven GAM stages from public interface through formula specification, basis construction, compilation, fitting, fitted results, and prediction; the fitting stage contains repeated coefficient and smoothing updates
   :class: scientific-figure

.. raw:: html
   :file: _includes/architecture_atlas.html

Basis construction is developed in
:doc:`statistical_reference/smooths/theory`. The inner coefficient solve and
the outer smoothing criterion are developed in
:doc:`statistical_reference/selection/theory`.

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
  The :ref:`custom-model guide <howto-user-guide-custom-models>` explains feature
  blocks, output width, and named contribution tensors.
* **Neural objective or distribution:** put target encoding, per-row loss,
  prediction semantics, and metrics in ``neural/objectives/`` or
  ``neural/families/``. Keep them independent of the forward architecture.
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
product. Finite coefficients are also distinct from convergence. A finite
coefficient vector is not a convergence claim. Convergence must be assessed
separately from finite output.

Executed examples test the public workflow and preserve useful outputs.
Notebook records identify source content, package versions, and code hashes;
the structural checker verifies links, anchors, navigation, and saved execution
records. These checks complement numerical tests: a runnable example can
still contain incorrect logic or misleading interpretation. Build and execution
commands have one home in :doc:`contributing`.

Detailed evidence
-----------------

* Focused statistical comparisons live in the `statistical parity tests
  <https://github.com/OpenTabular/NAMpy/tree/main/tests/parity>`_, smooth,
  optimization, diagnostics, and SCAM test directories.
* Neural task and architecture checks use small fitting budgets as interface
  evidence, not comparative performance benchmarks.
* User-facing restrictions belong with each method and family page.
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
These focused tests provide evidence beyond the small main and task sweeps;
they do not imply exhaustive coverage of every interaction
order, sparsity configuration, or objective combination.

Statistical extension scope
---------------------------

Implemented extensions cover fixed-smoothing expectiles and quantiles, joint
location/scale families, simultaneous linear-functional inference, piecewise
exponential survival data and exposure-history designs. Supported inputs and
limitations are defined in their reference pages.

Explicit ownership at public boundaries
---------------------------------------

Additive decompositions contain actual predictor coordinates only. ``link`` and
all contributions have shape ``(n, K)`` and ``predictor_names`` names those K
predictors. Response predictions belong to estimator methods; natural parameters
and probability queries belong to ``DistributionPrediction``. These widths may
differ: BEINF has four predictors and five natural parameters, while multinomial
models have K predictors and K+1 probabilities. No reference column is inserted
into an additive decomposition.

``nampy.diagnostics`` owns numerical preparation of distributional effects.
Backend adapters evaluate hypothetical predictor arrays through
``summarize_predictors``. Renderers consume prepared numerical data and do not
inspect neural objectives or import Torch. Conditional distribution spread and
between-fit variability remain separate quantities.

Neural constructor configuration uses explicit architecture, objective, training
and preprocessing records. Training preparation, route selection, gradient
execution and native/fixed execution have separate owners. The published neural
prediction module contains no Lightning training runtime. See
:ref:`howto-user-guide-training` for configuration and refit semantics.

The statistical fitting core returns ``FitCoreSolution.fit_result`` and
``FitCoreSolution.fit_state`` explicitly. Solver dictionaries are assembled at the
private solver boundary; there is no dictionary or attribute forwarding from the
solution record. Established asymmetric losses live in ``gam.losses``, functional
inference in ``gam.inference.functionals``, and lag/survival construction in
``gam.structured``. Statistical solver arithmetic and compilation stages retain
their mgcv semantics.
