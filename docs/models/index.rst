Neural architecture comparison
==============================

Start with :doc:`NAM <nam>` if you want an introduction to neural feature
effects. The other architectures make more specialized structural choices:
shared bases, sparse selection, interaction discovery, Gaussian-process
approximations, trees, transformers, quantiles, or splines. The comparisons
below describe those choices; they are not performance rankings.

For smoothness selection, statistical inference, and GAM diagnostics, use the
:doc:`statistical GAM guide <../user_guide/statistical_models>` instead.

Where to begin
--------------

Move beyond NAM only when the analysis supplies a reason for the added
structure.

.. list-table::
   :header-rows: 1
   :widths: 59 41

   * - What the analysis needs
     - Start with
   * - A straightforward neural additive model
     - :doc:`NAM <nam>`
   * - Basis functions shared across features and interactions
     - :doc:`NBM <nbm>`
   * - Sparse feature or term selection
     - :doc:`SNAM <snam>`
   * - Sparse interaction discovery
     - :doc:`SIAN <sian>`
   * - Additive Gaussian-process approximations
     - :doc:`GPNAM <gpnam>`
   * - Polynomial interaction structure
     - :doc:`SPAM <spam>` or :doc:`NBM-SPAM <nbm_spam>`

Architecture reference
----------------------

Each linked page explains when to use the model, its mathematical structure,
the variables on which a reported contribution may depend, supported
objectives, a runnable fit-and-plot workflow, important controls, and
interpretation limits.

.. list-table::
   :header-rows: 1
   :widths: 27 73

   * - Model
     - Representation
   * - :doc:`LinReg <linreg>`
     - Additive linear effects trained through the neural estimator interface
   * - :doc:`NAM <nam>` :sup:`3`
     - One neural network per feature, with optional explicit interactions
   * - :doc:`SNAM <snam>` :sup:`4`
     - NAM feature networks with group-lasso sparsity
   * - :doc:`SIAN <sian>` :sup:`5`
     - Sparse higher-order terms selected through feature-interaction detection
   * - :doc:`GPNAM <gpnam>` :sup:`6`
     - Random Fourier feature approximations to additive Gaussian processes
   * - :doc:`IGANN <igann>` :sup:`7`
     - Sparse linear initialization followed by feature-wise ELM boosting
   * - :doc:`NBM <nbm>` :sup:`8`
     - Learned basis functions shared across features and interactions
   * - :doc:`SPAM <spam>` :sup:`9`
     - Low-rank polynomial effects for scalable higher-order interactions
   * - :doc:`NBM-SPAM <nbm_spam>` :sup:`8, 9`
     - Learned NBM concept scores combined by SPAM polynomial heads
   * - :doc:`TreeNAM <treenam>` :sup:`3, 10`
     - One differentiable decision tree per main effect or explicit interaction
   * - :doc:`EnsembleTreeNAM <ensemble_treenam>` :sup:`3, 10`
     - A jointly trained ensemble of TreeNAM components
   * - :doc:`NodeGAM <nodegam>` :sup:`10`
     - Additive ensembles of differentiable oblivious trees
   * - :doc:`NATT <natt>` :sup:`11`
     - Additive tabular-transformer representations with optional interactions
   * - :doc:`NAMformer <namformer>` :sup:`12`
     - Transformer representations designed to recover marginal feature effects
   * - :doc:`QNAM <qnam>` :sup:`3, 13`
     - Additive quantile functions constrained to remain ordered
   * - :doc:`SplineNAM <spline_nam>` :sup:`1, 2`
     - Learnable cubic-spline feature and interaction functions

``Regressor``, ``Classifier``, and ``LSS`` suffixes identify the estimator
task, for example ``NAMRegressor``, ``NAMClassifier``, and ``NAMLSS``. QNAM is
distributional-only, SplineNAM is regression-only, and IGANN currently supports
main effects but not explicit interaction terms. The :doc:`../api/models`
reference lists every public estimator class.

The :doc:`../tutorials/nam_nbm` compares two neural representations on the
same data and plotting scale. The :doc:`../notebooks/03_neural_models`
provides a broader executable architecture survey.

.. toctree::
   :maxdepth: 1
   :hidden:

   linreg
   nam
   snam
   sian
   gpnam
   igann
   nbm
   spam
   nbm_spam
   treenam
   ensemble_treenam
   nodegam
   natt
   namformer
   qnam
   spline_nam

References
----------

1. Hastie and Tibshirani (1986). `Generalized Additive Models
   <https://doi.org/10.1214/ss/1177013604>`_.
2. Wood (2011). `Fast stable restricted maximum likelihood and marginal
   likelihood estimation of semiparametric generalized linear models
   <https://doi.org/10.1111/j.1467-9868.2010.00749.x>`_.
3. Agarwal et al. (2021). `Neural Additive Models: Interpretable Machine
   Learning with Neural Nets <https://arxiv.org/abs/2004.13912>`_.
4. Xu et al. (2022). `Sparse Neural Additive Model: Interpretable Deep
   Learning with Feature Selection via Group Sparsity
   <https://arxiv.org/abs/2202.12482>`_.
5. Enouen and Liu (2023). `Sparse Interaction Additive Networks via Feature
   Interaction Detection and Sparse Selection
   <https://arxiv.org/abs/2209.09326>`_.
6. Zhang, Barr, and Paisley (2024). `Gaussian Process Neural Additive Models
   <https://arxiv.org/abs/2402.12518>`_.
7. Kraus et al. (2024). `Interpretable generalized additive neural networks
   <https://doi.org/10.1016/j.ejor.2023.06.032>`_.
8. Radenovic, Dubey, and Mahajan (2022). `Neural Basis Models for
   Interpretability <https://arxiv.org/abs/2205.14120>`_.
9. Dubey, Radenovic, and Mahajan (2022). `Scalable Interpretability via
   Polynomials <https://arxiv.org/abs/2205.14108>`_.
10. Chang, Caruana, and Goldenberg (2022). `NODE-GAM: Neural Generalized
    Additive Model for Interpretable Deep Learning
    <https://arxiv.org/abs/2106.01613>`_.
11. Thielmann et al. `Neural Additive Tabular Transformer Networks
    <https://openreview.net/forum?id=TdJ7lpzAkD>`_.
12. Thielmann, Reuter, and Säfken (2025). `Beyond Black-Box Predictions:
    Identifying Marginal Feature Effects in Tabular Transformer Networks
    <https://arxiv.org/abs/2504.08712>`_.
13. Koenker and Bassett (1978). `Regression Quantiles
    <https://doi.org/10.2307/1913643>`_.
