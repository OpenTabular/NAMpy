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
