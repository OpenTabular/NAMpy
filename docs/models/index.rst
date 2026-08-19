Models
======

NAMpy provides a comprehensive suite of interpretable and high-performance models
for tabular data. Most models are available in three variants: regression, classification,
and distributional regression (LSS). QNAM is specialized for distributional
regression; TreeNAM, EnsembleTreeNAM, and SNAM support all three task types.

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
   * - GPNAM
     - Gaussian Process NAM
     - ✓ Yes
     - Uncertainty Quantification
   * - NBM
     - Neural Basis Model
     - ✓ Partial
     - Complex Non-linearities
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
:math:`\mathcal{I}` is used when ``interaction_degree >= 2``.

.. code-block:: python

   from nampy.models import NAMRegressor
   
   model = NAMRegressor(
       layer_sizes=[128, 128, 32],
       numerical_preprocessing="ple"
   )

**GPNAM (Gaussian Process NAM)**

Extends NAM with Gaussian processes for uncertainty quantification.

.. math::

   \phi(x) = \sqrt{\frac{2}{M}}
   \left[\cos\left(\frac{z_m x_j}{\ell} + c_{m,j}\right)\right]_{j=1..d,\,m=1..M},
   \quad f(x) = \phi(x)^\top w

This model uses random Fourier features (RFF) with kernel width :math:`\ell` and
learns a linear predictor over the RFF mapping.

.. code-block:: python

   from nampy.models import GPNAMRegressor
   
   model = GPNAMRegressor()

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
   
   model = NodeGAMRegressor()

Basis Function Models
~~~~~~~~~~~~~~~~~~~~~

**NBM (Neural Basis Model)**

Uses learned basis functions for feature transformations.

.. math::

   h(x) = \Big[B_{S,k}(x_S)\Big]_{S \in \mathcal{S},\,k=1..K}, \quad
   f(x) = W\,g(h(x)) + b

For each n-ary subset :math:`S`, the basis network outputs :math:`K` basis
responses that are combined by a grouped featurizer :math:`g` and a final linear
layer.

.. code-block:: python

   from nampy.models import NBMRegressor
   
   model = NBMRegressor()

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
   
   model = NAMLSS()
   model.fit(X, y, family="normal", max_epochs=150)

Model Reference
---------------

For full constructor arguments, keyword arguments, and hyperparameters for each
model class, see the :doc:`../api/models` page.

Complete API Reference
----------------------

For detailed API documentation, see :doc:`../api/models`.
