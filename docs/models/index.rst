Models
======

NAMpy provides a comprehensive suite of interpretable and high-performance models
for tabular data. Most models are available in three variants: regression, classification,
and distributional regression (LSS). QNAM is specialized for distributional
regression; NBM, SPAM, NBM-SPAM, TreeNAM, EnsembleTreeNAM, and SNAM support
all three task types.

Model Overview
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 20 20

   * - Model
     - Description
     - Interpretable
     - Best For
   * - NAM
     - Neural Additive Model
     - ✓ Yes
     - Interpretability + Performance
   * - SIAN
     - Sparse Interaction Additive Network
     - ✓ Yes
     - Automatically selected higher-order interactions
   * - GPNAM
     - Fixed-RFF Gaussian Process NAM
     - ✓ Yes
     - Fast smooth additive regression
   * - IGANN
     - Linear initialization with boosted ELM shape functions
     - ✓ Yes
     - Fast, smooth, optionally sparse additive fitting
   * - NBM
     - Neural Basis Model
     - ✓ Yes
     - Shared learned basis functions
   * - SPAM
     - Scalable Polynomial Additive Model
     - ✓ Yes
     - Low-rank polynomial effects
   * - NBM-SPAM
     - Neural bases with polynomial interactions
     - ✓ Yes
     - Learned concepts and higher-order structure
   * - NATT
     - Neural Attentive Transformer
     - ✗ No
     - Feature Interactions
   * - NAMformer
     - Transformer-based NAM
     - ✓ Partial
     - Balance of Both
   * - QNAMLSS
     - Quantile NAM (distributional-only)
     - ✓ Yes
     - Quantile Regression
   * - TreeNAM
     - Tree-based NAM
     - ✓ Yes
     - Categorical Features
   * - SNAM
     - Sparse NAM
     - ✓ Yes
     - Feature Selection
   * - NodeGAM
     - Node-based GAM
     - ✓ Yes
     - Large Datasets
   * - LinReg
     - Neural Linear Regression
     - ✓ Yes
     - Linear Relationships

Model Families
--------------

Additive Models (Interpretable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These models decompose predictions into feature-level contributions:

**NAM (Neural Additive Model)**

The foundational interpretable model. Each feature's contribution is modeled
independently using a neural network.

.. math::

   f(x) = \beta_0 + \sum_{i=1}^{d} f_i(x_i) + \sum_{S \in \mathcal{I}} f_S(x_S)

Where :math:`f_i` is a feature-specific MLP, and the optional interaction set
:math:`\mathcal{I}` is used when ``interaction_degree >= 2``. Architectures
using the shared interaction scaffold also accept an explicit sparse set such
as ``interactions=(("age", "income"),)``; specifying both forms is an error.

.. code-block:: python

   from nampy.models import NAMRegressor
   
   model = NAMRegressor(
       layer_sizes=[128, 128, 32],
       numerical_preprocessing="ple"
   )

NAM capabilities are composable rather than tied to an architecture preset.
For example, ExU feature networks, train-cardinality-dependent widths, and
feature-output regularization can be enabled independently:

.. code-block:: python

   import torch.nn as nn

   model = NAMRegressor(
       feature_layer="exu",       # also "linear" or "centered_relu"
       activation=nn.ReLU,         # activation for later linear hidden layers
       adaptive_width=True,
       num_basis_functions=1000,
       units_multiplier=2,
       output_regularization=1e-4,
       l2_regularization=1e-6,
       feature_output_bias=False,
   )

``feature_widths={"feature_name": width}`` overrides the first hidden width
for selected main effects. ``regularize_interactions=True`` extends the output
penalty to configured interaction terms.

**SIAN (Sparse Interaction Additive Network)**

SIAN first fits a reference ReLU MLP, scores candidate feature subsets with
Archipelago inclusion/removal contrasts, applies fractional-heredity frontier
selection, and then trains an additive network containing the selected terms.
Selection operates on logical source features, so transformed columns from a
one-hot or basis expansion are perturbed together. Main effects are always
included; sparse selection controls the higher-order term set.

.. code-block:: python

   from nampy.models import SIANRegressor

   model = SIANRegressor(
       max_interaction_order=3,
       interaction_thresholds={2: 0.10, 3: 0.05},
       threshold_mode="fraction",
       heredity_fraction=0.5,
   )
   model.fit(X_train, y_train)

   selected = model.selected_interactions_
   scores = model.interaction_selection_table()

``threshold_mode="fraction"`` retains the requested highest-scoring fraction
at each order. ``"absolute"`` applies raw score cutoffs and ``"quantile"``
applies empirical quantiles. Supplying ``interactions=(("age", "income"),)``
bypasses reference fitting and discovery.

The SIAN preset uses the paper's ``[16, 12, 8]`` ReLU shape networks,
Adagrad learning rate ``5e-3``, and L1 coefficient ``5e-5``. These remain
ordinary constructor or fit controls rather than a separate training engine.

The default ``execution_mode`` is ``"block_masked"``. Call
``model.compress_terms()`` for independent shape subnetworks and
``model.block_mask_terms()`` to reconstruct the parallel form without changing
predictions. The same
architecture declaration generates ``SIANRegressor``, ``SIANClassifier``, and
``SIANLSS`` through NAMpy's objective registry.

For multi-output regression, multiclass classification, or multivariate LSS,
``selection_output_index`` chooses the scalar target used by discovery.
``residual_network=True`` adds the upstream maximal-update-parameterized
unrestricted contribution named
``"residual"``; the result remains decomposed, but it is no longer purely
additive in individual source features.

Higher-order terms appear in ``predict_components()``, ``explain_terms()``,
and ``interaction_importance()``. Explanation tables retain one ``value_N``
column per interaction dimension, while ``plot_interactions()`` renders
conditioned heatmap slices above order two.

**GPNAM (Gaussian Process NAM)**

Uses a fixed random-Fourier-feature approximation to one RBF-kernel shape
function per scalar input. Only the additive RFF coefficients and intercept
are learned. This is a point-estimation model; it does **not** compute a GP
posterior covariance or epistemic predictive uncertainty.

.. math::

   \phi(x) = \sqrt{\frac{2}{M}}
   \left[\cos\left(\frac{z_m x_j}{\ell} + c_{m,j}\right)\right]_{j=1..d,\,m=1..M},
   \quad f(x) = \phi(x)^\top w

The default quasi-random construction uses an inverse-normal frequency grid
and a separately permuted uniform phase grid for every input. Numerical values
are passed to GPNAM without generic PLE expansion. By default, feature-specific
kernel widths are fitted as ``std(x) / 24`` on training rows only.

Ordinary regression uses the reference conjugate-gradient ridge solve with an
unpenalized intercept. ``ridge=0.05`` matches the released Python and MATLAB
implementations. Set ``solver="gradient"`` to use the shared Lightning engine;
classification and LSS always use that engine. ``GPNAMLSS`` is a NAMpy
distributional extension and represents aleatoric distribution parameters,
not a GP posterior.

.. code-block:: python

   from nampy.models import GPNAMRegressor
   
   model = GPNAMRegressor(
       kernel_width="auto",
       rff_num_feat=100,
       solver="cg",
       ridge=0.05,
       rff_random_state=7,
   )
   model.fit(X_train, y_train)

   # Fixed-basis and solver diagnostics
   Phi = model.basis_transform(X_test, batch_size=1024)
   metadata = model.basis_metadata()
   complexity = model.model_complexity()

The GP-NA2M extension adds selected two-dimensional RFF interaction terms:

.. code-block:: python

   model = GPNAMRegressor(
       interactions=(("age", "income"), ("debt", "assets")),
       rff_num_feat=100,
   )

``interaction_degree=2`` instead constructs every pair. Explicit interactions
are preferable when the full quadratic set would be unnecessarily large.

**IGANN (Interpretable Generalized Additive Neural Network)**

IGANN starts from an L1-regularized linear model and sequentially adds shallow
extreme learning machines (ELMs). Every numerical feature owns ``n_hid`` fixed
random hidden units, categorical features remain reference-coded linear terms,
and each stage solves only its output coefficients by ridge regression. The
result stays exactly additive while allowing a linear relationship to bend only
when the boosted residuals justify it.

.. math::

   f(x) = \beta_0 + \sum_j \beta_j x_j
          + \sum_{t=1}^{T} \eta \sum_j
            w_{t,j}^{\mathsf T}\sigma(a_{t,j}x_j)

The upstream Newton-style pseudo-response and Hessian scaling are implemented
through NAMpy's architecture-native training contract. Therefore
``n_estimators`` and ``early_stopping`` are IGANN constructor controls rather
than Lightning epochs. The shared preprocessing pipeline remains fitted on
training rows only, avoiding the validation leakage present in the released
standalone fit path.

.. code-block:: python

   from nampy.models import IGANNRegressor

   model = IGANNRegressor(
       n_hid=10,
       n_estimators=500,
       boost_rate=0.1,
       init_reg=1.0,
       elm_scale=1.0,
       elm_alpha=1.0,
       early_stopping=50,
   )
   model.fit(X_train, y_train, random_state=7)

   components = model.predict_components(X_test, batch_size=2048)
   history = model.training_history()
   metadata = model.basis_metadata()
   complexity = model.model_complexity()

For regression and binary classification, ``solver="auto"`` selects the
released stagewise optimizer. Multiclass classification and ``IGANNLSS`` use
the shared gradient objective engine over the complete fixed ELM basis because
the released second-order updates do not define those losses. This is an
explicit NAMpy extension: set ``solver="gradient"`` to request it for any task,
or ``solver="native"`` to require the reference optimizer and reject an
unsupported objective.

.. code-block:: python

   from nampy.models import IGANNLSS

   model = IGANNLSS(
       family="normal",
       n_hid=10,
       n_estimators=100,
   )
   model.fit(X_train, y_train, max_epochs=100)

IGANN-Sparse applies grouped best-subset selection to the first ELM's nonlinear
feature blocks before fitting the linear initialization and boosting sequence:

.. code-block:: python

   model = IGANNRegressor(sparse=7)

This native-training option requires ``abess>=0.4.5`` (install
``nampy[igann-sparse]``). With
``sparse=0``—the default—ABESS is neither imported nor required. Selected
atomic terms are available as ``selected_features_``. Pairwise interactions
are intentionally absent: the IGANN-Sparse paper lists them as future work and
the current stable upstream model does not implement them.

For bagging, use the generic ensemble rather than an IGANN-specific wrapper:

.. code-block:: python

   from nampy.models import IGANNRegressor, NeuralEnsemble

   ensemble = NeuralEnsemble(
       IGANNRegressor(n_estimators=300),
       n_estimators=5,
       bootstrap=True,
   )
   ensemble.fit(X_train, y_train)

**TreeNAM**

Uses one differentiable neural decision tree per feature, with optional
interaction trees.

.. math::

   f(x) = \sum_{i=1}^{d} f_i(x_i), \quad
   f_i(x_i) = \sum_{m=1}^{M} \eta \, T_{i,m}(x_i)

Each feature contribution is produced by a soft neural decision tree; routing
can be made hard during evaluation.

.. code-block:: python

   from nampy.models import TreeNAMRegressor
   
   model = TreeNAMRegressor()

**SNAM (Sparse NAM)**

Applies sparsity constraints for automatic feature selection.

.. math::

   f(x) = \beta_0 + \sum_{i=1}^{d} f_i(x_i)
   + \sum_{S \in \mathcal{I}} f_S(x_S), \qquad
   \Omega(\theta) = \lambda \sum_j \lVert\theta_j\rVert_2

SNAM reuses NAM feature subnetworks and applies a group-lasso penalty to each
subnetwork's trainable parameter vector.

.. code-block:: python

   from nampy.models import SNAMRegressor
   
   model = SNAMRegressor()

**NodeGAM**

Efficient node-based generalized additive model.

.. math::

   f(x) = \sum_{t=1}^{T} g_t(x_{S_t}), \quad |S_t| \in \{1,2\}

Each oblivious tree :math:`g_t` selects one feature (GAM) or a pair of features
(GA2M) and the outputs are summed across trees and layers.

.. code-block:: python

   from nampy.models import NodeGAMRegressor

   model = NodeGAMRegressor(
       selector_activation="entmax15",  # or "sparsemax"
       bin_activation="entmoid15",      # or "sparsemoid"
       l1_interactions=0.0,
       l2_interactions=1e-4,
   )
   model.fit(
       X_train,
       y_train,
       pretrain_epochs=5,       # optional masked reconstruction
       average_checkpoints=True,
       n_last_checkpoints=5,
   )

The neural estimators and GAM adapters share ``explain_terms()``,
``term_importance()``, and ``interaction_importance()``. Quantile input maps
are available to every neural estimator with
``numerical_preprocessing="quantile"``. Published PreTab 0.0.3 owns the
quantile transform defaults and does not expose separate ``quantile_*``
constructor controls.

Term contributions can be centered without changing predictions. By default
they are centered on the explained rows; ``reference_X`` can supply a separate
reference population:

.. code-block:: python

   components = model.predict_components(X_test, center=True)
   importance = model.term_importance(
       X_test, center=True, reference_X=X_train
   )

All neural objectives, including LSS distributions, accept fit-time
``sample_weight``. Classifiers additionally accept ``class_weight``;
``sampling_strategy="balanced"`` enables inverse-frequency sampling while
keeping sampling separate from loss weights.

Independent fitted estimators can be combined with ``NeuralEnsemble``:

.. code-block:: python

   from nampy.models import NAMRegressor, NeuralEnsemble

   ensemble = NeuralEnsemble(NAMRegressor(), n_estimators=5, n_jobs=1)
   ensemble.fit(X_train, y_train, max_epochs=100)
   uncertainty = ensemble.predict_component_uncertainty(X_test)

Basis Function Models
~~~~~~~~~~~~~~~~~~~~~

**NBM (Neural Basis Model)**

Uses learned basis functions for feature transformations.

.. math::

   h(x) = \Big[B_{S,k}(x_S)\Big]_{S \in \mathcal{S},\,k=1..K}, \quad
   f(x) = W\,g(h(x)) + b

For each n-ary subset :math:`S`, the basis network outputs :math:`K` basis
responses that are combined by a grouped featurizer :math:`g` and a final linear
layer. The default hidden topology, ReLU/BatchNorm behavior, basis dropout,
and grouped ``Conv1d`` featurizer match the released NBM definition. Set
``sparse=True`` to use its sentinel-driven active-tuple execution path;
``nary_ignore_input`` controls the ignored value globally or by order.

.. code-block:: python

   from nampy.models import NBMRegressor
   
   model = NBMRegressor(nary=[1, 2])

The default preprocessor supplies one scalar transformed column per concept.
PLE/spline expansions remain available explicitly, but each resulting column
then becomes its own NBM concept.

**SPAM (Scalable Polynomial Additive Model)**

SPAM adds low-rank homogeneous polynomial blocks by degree. ``ranks[i]`` is
the rank for degree ``i + 2``. Tensor and basis-L1 penalties are controlled
independently.

.. code-block:: python

   from nampy.models import SPAMRegressor

   model = SPAMRegressor(
       ranks=[200, 50],
       regularization_scale=1e-5,
   )

After fitting, ``model.local_term_importance(X, top_k=10)`` expands the
degree-specific low-rank representation into the upstream-style local unary
and distinct-variable polynomial terms.

**NBM-SPAM**

NBM-SPAM first learns unary shared-basis scores, then combines one score block
linearly and the remaining blocks through degree-specific SPAM heads. As in
the reference implementation, the NBM stage is restricted to unary terms;
the polynomial heads create the higher-order structure.

.. code-block:: python

   from nampy.models import NBMSPAMRegressor

   model = NBMSPAMRegressor(
       num_bases=100,
       ranks=[200],
   )

Attention-Based Models
~~~~~~~~~~~~~~~~~~~~~~

**NATT (Neural Attentive Tabular Transformer)**

Transformer architecture with attention mechanisms.

.. math::

   f(x) = \beta_0 + \sum_{i=1}^{d_\text{num}} f_i(x_i)
   + g\!\left(\mathrm{Transformer}(E(x_\text{cat}))\right)
   + \sum_{S \in \mathcal{I}} f_S(x_S)

Numerical features use per-feature subnetworks, while categorical features are
embedded and passed through a transformer encoder and MLP head :math:`g`.

.. code-block:: python

   from nampy.models import NATTRegressor
   
   model = NATTRegressor()

**NAMformer**

Combines NAM's interpretability with transformer architecture.

.. math::

   h = \mathrm{Transformer}(E(x)), \quad
   f(x) = g(h_{\text{[CLS]}}) + \sum_{i=1}^{d} f_i(e_i) + \sum_{S \in \mathcal{I}} f_S(e_S)

Each token embedding :math:`e_i` contributes through a linear head :math:`f_i`,
while the [CLS] token is mapped by :math:`g` and summed with interactions.

.. code-block:: python

   from nampy.models import NAMformerRegressor
   
   model = NAMformerRegressor()

Specialized Models
~~~~~~~~~~~~~~~~~~

**QNAM (Quantile NAM)**

Designed specifically for quantile regression.

.. math::

   q_{\tau_k}(x) = \beta_0 + \sum_{i=1}^{d} f_i(x_i) + \sum_{S \in \mathcal{I}} f_S(x_S),
   \quad k = 1,\dots,K

The model outputs a set of quantile functions :math:`q_{\tau_k}` for requested
quantile levels :math:`\tau_k`.

.. code-block:: python

   from nampy.models import QNAMLSS
   
   model = QNAMLSS()
   model.fit(X, y, distributional_kwargs={"quantiles": [0.1, 0.5, 0.9]})

**LinReg (Neural Linear Regression)**

Neural network implementation of linear regression.

.. math::

   f(x) = \beta_0 + \sum_{i=1}^{d} w_i^\top x_i

Each feature is mapped by a linear layer and summed with an intercept term.

.. code-block:: python

   from nampy.models import LinRegRegressor
   
   model = LinRegRegressor()

Task Variants
-------------

Most models are available in three variants:

Regression
~~~~~~~~~~

For continuous target variables:

.. code-block:: python

   from nampy.models import NAMRegressor, GPNAMRegressor, NBMRegressor
   # etc.

Classification
~~~~~~~~~~~~~~

For categorical target variables:

.. code-block:: python

   from nampy.models import NAMClassifier, GPNAMClassifier, NBMClassifier
   # etc.

Distributional Regression (LSS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For modeling full probability distributions:

.. code-block:: python

   from nampy.models import NAMLSS, GPNAMLSS, NBMLSS, NodeGAMLSS
   # etc.
   
   model = NAMLSS(family="normal")
   model.fit(X, y, max_epochs=150)

Model Reference
---------------

For full constructor arguments, keyword arguments, and hyperparameters for each
model class, see the :doc:`../api/models` page.

Complete API Reference
----------------------

For detailed API documentation, see :doc:`../api/models`.
