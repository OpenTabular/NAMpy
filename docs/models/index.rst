Models and Architectures
========================

NAMpy provides statistical and neural models through a common estimator API.
The models differ in how they represent feature effects, select interactions,
and estimate their parameters. Use the table below to choose an appropriate
starting point.

Where to begin
--------------

These are structural starting points, not performance rankings.

.. list-table::
   :header-rows: 1
   :widths: 59 41

   * - What the analysis needs
     - Start with
   * - Statistical smoothness selection, standard errors, and diagnostics
     - :doc:`GAM <gam>`
   * - A straightforward neural additive model
     - :doc:`NAM <nam>`
   * - Basis functions shared across features and interactions
     - :doc:`NBM <nbm>`
   * - Joint identifiable spline and neural composition
     - :doc:`SemiStructured <semistructured>`
   * - Sparse feature or term selection
     - :doc:`SNAM <snam>`
   * - Sparse interaction discovery
     - :doc:`SIAN <sian>`
   * - Additive Gaussian-process approximations
     - :doc:`GPNAM <gpnam>`
   * - Polynomial interaction structure
     - :doc:`SPAM <spam>` or :doc:`NBM-SPAM <nbm_spam>`

Prediction and explanation contracts
------------------------------------

Task support comes from the architecture registry. Explanation support is a
separate property: reconstructing a predictor does not imply that every term
is a marginal effect, or that estimated effects are identifiable and stable.
For statistical GAM inference and its bounds, see :doc:`gam`.

Neural architecture matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~

The matrix below covers neural architectures; the statistical GAM reference
is :doc:`gam`. Classification components describe actual additive logits;
probabilities are queried separately. Dependence metadata distinguishes main,
interaction, contextual and global contributions. Intervals and uncertainty require their own assumptions and
calibration checks; this table is a structural chooser, not a leaderboard.

SIAN's optional unrestricted residual network changes the dependence contract.
SplineNAM's interactions use projected design terms rather than neural
interaction subnetworks. Follow the architecture links for those restrictions.

.. include:: /_generated/reference/model_capabilities.inc

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
   * - :doc:`GAM <gam>` :sup:`1, 2`
     - Penalized regression splines with automatic smoothness selection and
       statistical inference
   * - :doc:`LinReg <linreg>`
     - Additive linear effects trained through the neural estimator interface
   * - :doc:`NAM <nam>` :sup:`3`
     - One neural network per feature, with optional explicit interactions
   * - :doc:`SNAM <snam>` :sup:`4`
     - NAM feature networks with group-lasso sparsity
   * - :doc:`SIAN <sian>` :sup:`5, 16`
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
   * - :doc:`NodeGAM <nodegam>` :sup:`10, 14`
     - Additive ensembles of differentiable oblivious trees; NodeGAMLSS models distribution parameters
   * - :doc:`NATT <natt>` :sup:`11`
     - Additive tabular-transformer representations with optional interactions
   * - :doc:`NAMformer <namformer>` :sup:`12`
     - Feature-specific marginal heads plus a global transformer contribution
   * - :doc:`QNAM <qnam>` :sup:`3, 13`
     - Additive quantile functions constrained to remain ordered
   * - :doc:`SplineNAM <spline_nam>`
     - Learnable cubic-spline feature and interaction functions

The neural distributional framework follows Thielmann et al. (2024), reference 15.
TreeNAM and EnsembleTreeNAM adapt additive and differentiable-tree components;
the cited papers do not introduce those exact NAMpy implementations.

``Regressor``, ``Classifier``, and ``LSS`` suffixes identify the estimator
task, for example ``NAMRegressor``, ``NAMClassifier``, and ``NAMLSS``. QNAM is
distributional-only, SplineNAM is regression-only, and IGANN currently supports
main effects but not explicit interaction terms. The :doc:`../api/models`
reference lists every public estimator class.

The :doc:`neural architecture comparison <../notebooks/03_neural_models>` compares independent feature networks with shared bases and examines marginal and global contributions.

.. toctree::
   :maxdepth: 1
   :hidden:

   gam
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
   semistructured

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
11. Thielmann et al. (2024). `Interpretable Additive Tabular Transformer Networks
    <https://openreview.net/forum?id=TdJ7lpzAkD>`_. Transactions on Machine Learning Research.
12. Thielmann, Reuter, and Säfken (2025). `Beyond Black-Box Predictions:
    Identifying Marginal Feature Effects in Tabular Transformer Networks
    <https://arxiv.org/abs/2504.08712>`_.
13. Koenker and Bassett (1978). `Regression Quantiles
    <https://doi.org/10.2307/1913643>`_.

14. De, Thielmann, and Säfken (2024), `NODE-GAMLSS: Interpretable Uncertainty
    Modelling via Deep Distributional Regression
    <https://openreview.net/forum?id=sWvTLlRkzy>`_. NeurIPS 2024 Workshop on Bayesian
    Decision-Making and Uncertainty.
15. Thielmann, Kruse, Kneib, and Säfken (2024), `Neural Additive Models for
    Location Scale and Shape: A Framework for Interpretable Neural Regression
    Beyond the Mean <https://proceedings.mlr.press/v238/frederik-thielmann24a.html>`_.
    AISTATS, PMLR 238:1783–1791.
16. Tsang, Rambhatla, and Liu (2020), `How does This Interaction Affect Me?
    Interpretable Attribution for Feature Interactions
    <https://proceedings.neurips.cc/paper_files/paper/2020/hash/443dec3062d0286986e21dc0631734c9-Abstract.html>`_.
    NeurIPS 33.
