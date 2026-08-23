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
     - :doc:`../notebooks/01_gam`

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
     - :doc:`../notebooks/02_linreg`
   * - NAM
     - Independent neural shape functions and optional explicit interactions
     - :doc:`../notebooks/03_nam`
   * - SNAM
     - Sparse additive fitting and feature selection
     - :doc:`../notebooks/04_snam`
   * - SIAN
     - Data-driven sparse higher-order interaction discovery
     - :doc:`../notebooks/05_sian`
   * - GPNAM
     - Smooth fixed random-Fourier-feature additive effects
     - :doc:`../notebooks/06_gpnam`
   * - IGANN
     - Fast boosted ELM shape functions initialized by a sparse linear model
     - :doc:`../notebooks/07_igann`
   * - NBM
     - Shared learned basis functions for unary or n-ary concepts
     - :doc:`../notebooks/08_nbm`
   * - SPAM
     - Low-rank polynomial additive effects
     - :doc:`../notebooks/09_spam`
   * - NBM-SPAM
     - Polynomial heads over learned NBM concept scores
     - :doc:`../notebooks/10_nbm_spam`
   * - TreeNAM
     - Piecewise additive effects learned with differentiable trees
     - :doc:`../notebooks/11_treenam`
   * - NodeGAM
     - Additive ensembles of differentiable oblivious trees
     - :doc:`../notebooks/13_nodegam`
   * - NATT
     - Attention-based tabular modeling when strict additivity is not required
     - :doc:`../notebooks/14_natt`
   * - NAMformer
     - Transformer-style representations with additive term extraction
     - :doc:`../notebooks/15_namformer`
   * - QNAM
     - Ordered conditional quantiles through distributional regression
     - :doc:`../notebooks/16_qnam`
   * - SplineNAM
     - Neural additive effects represented by learnable spline bases
     - :doc:`../notebooks/17_spline_nam`

Ensembling
----------

``EnsembleTreeNAM`` is the jointly trained multi-TreeNAM architecture; it is
documented as a TreeNAM variant in :doc:`../notebooks/12_ensemble_treenam`.
``NeuralEnsemble`` is the independent bootstrap/seed ensemble wrapper that can
be applied to supported neural estimators; see
:doc:`../notebooks/18_neural_ensemble`.

Task availability
-----------------

The notebook for each architecture constructs every supported estimator
surface: regression, classification, and/or distributional regression. The
:doc:`../api/models` page is the authoritative list of public estimator
classes.
