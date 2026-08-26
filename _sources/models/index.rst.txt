Model Catalog
=============

This page helps choose a model. Mathematical derivations, supported task
variants, model-specific controls, interpretation examples, limitations, and
primary references live in the :doc:`../notebook_tutorials`. Constructor and
method signatures live in :doc:`../api/index`.

Statistical model
-----------------

.. list-table::
   :header-rows: 1
   :widths: 18 38 44

   * - Model
     - Main inductive bias
     - Start here
   * - GAM
     - Penalized regression splines with ``mgcv``-aligned fitting and inference
     - :doc:`../notebooks/02_complete_gam_guide`

Neural architectures
--------------------

.. list-table::
   :header-rows: 1
   :widths: 18 48 34

   * - Model
     - Best fit for
     - Tutorial
   * - LinReg
     - An additive linear baseline
     - :doc:`../notebooks/03_neural_additive_model_zoo`
   * - NAM
     - Independent neural shape functions and optional explicit interactions
     - :doc:`../notebooks/03_neural_additive_model_zoo`
   * - SNAM
     - Sparse additive fitting and feature selection
     - :doc:`../notebooks/03_neural_additive_model_zoo`
   * - SIAN
     - Data-driven sparse higher-order interaction discovery
     - :doc:`../notebooks/03_neural_additive_model_zoo`
   * - GPNAM
     - Smooth fixed random-Fourier-feature additive effects
     - :doc:`../notebooks/03_neural_additive_model_zoo`
   * - IGANN
     - Fast boosted ELM shape functions initialized by a sparse linear model
     - :doc:`../notebooks/03_neural_additive_model_zoo`
   * - NBM
     - Shared learned basis functions for unary or n-ary concepts
     - :doc:`../notebooks/03_neural_additive_model_zoo`
   * - SPAM
     - Low-rank polynomial additive effects
     - :doc:`../notebooks/03_neural_additive_model_zoo`
   * - NBM-SPAM
     - Polynomial heads over learned NBM concept scores
     - :doc:`../notebooks/03_neural_additive_model_zoo`
   * - TreeNAM
     - Piecewise additive effects learned with differentiable trees
     - :doc:`../notebooks/03_neural_additive_model_zoo`
   * - NodeGAM
     - Additive ensembles of differentiable oblivious trees
     - :doc:`../notebooks/03_neural_additive_model_zoo`
   * - NATT
     - Attention-based tabular modeling when strict additivity is not required
     - :doc:`../notebooks/03_neural_additive_model_zoo`
   * - NAMformer
     - Transformer-style representations with additive term extraction
     - :doc:`../notebooks/03_neural_additive_model_zoo`
   * - QNAM
     - Ordered conditional quantiles through distributional regression
     - :doc:`../notebooks/03_neural_additive_model_zoo`
   * - SplineNAM
     - Neural additive effects represented by learnable spline bases
     - :doc:`../notebooks/03_neural_additive_model_zoo`

Ensembling
----------

``EnsembleTreeNAM`` is the jointly trained multi-TreeNAM architecture and is
covered in :doc:`../notebooks/03_neural_additive_model_zoo`.
``NeuralEnsemble`` is the independent bootstrap/seed ensemble wrapper that can
be applied to supported neural estimators; see
:doc:`../notebooks/04_tasks_distributional_models_and_ensembles`.

Task availability
-----------------

The runtime sweeps in
:doc:`../notebooks/04_tasks_distributional_models_and_ensembles` construct and
fit each supported regression, classification, and distributional estimator
surface. The :doc:`../api/models` page is the authoritative list of public
estimator classes.
