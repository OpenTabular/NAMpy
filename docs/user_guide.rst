User Guide
==========

This guide provides comprehensive information on using NAMpy effectively.

.. toctree::
   :maxdepth: 2

   user_guide/preprocessing
   user_guide/pretab_compatibility
   user_guide/training
   user_guide/shape_constrained_gams
   user_guide/custom_models
   user_guide/interpretability

Model Selection
---------------

NAMpy provides multiple model architectures for different use cases:

Standard Models
~~~~~~~~~~~~~~~

**NAM (Neural Additive Model)**
   The foundational interpretable model. Best for when interpretability is crucial
   and you need feature-level attributions.

**Linear Regression (Neural)**
   A neural network implementation of linear regression. Useful as a baseline
   or when linear relationships are expected.

Advanced Models
~~~~~~~~~~~~~~~

**GPNAM (Gaussian Process NAM)**
   Uses fixed random Fourier features to learn smooth additive RBF-kernel shape
   functions with a convex regression objective. It is parameter-efficient and
   interpretable, but does not itself provide GP posterior uncertainty.

**IGANN**
   Initializes additively with a sparse linear model and then boosts fixed
   feature-wise ELM bases. It is useful when fast training, smooth shape
   functions, and optional nonlinear feature selection are priorities.

**SIAN**
   Detects interactions with a reference MLP and Archipelago, then trains a
   sparse additive model with selected interactions of configurable order.

**NBM (Neural Basis Model)**
   Learns shared basis functions for unary or n-ary concept tuples. Dense and
   sparse active-tuple execution are selected through configuration.

**SPAM and NBM-SPAM**
   SPAM learns low-rank polynomial effects. NBM-SPAM applies those polynomial
   heads to learned unary NBM scores.

**NATT (Neural Attentive Tabular Transformer)**
   Transformer-based architecture with attention mechanisms. Best for
   datasets with complex feature interactions.

**NAMformer**
   Combines NAM's interpretability with transformer architecture.
   Good balance between performance and interpretability.

Specialized Models
~~~~~~~~~~~~~~~~~~

**QNAM (Quantile NAM)**
   Designed for quantile regression. Use when you need to predict
   specific quantiles of the target distribution.

**TreeNAM**
   Integrates tree-based methods with NAM. Good for tabular data
   with categorical features.

**SNAM (Sparse NAM)**
   Applies sparsity constraints for feature selection. Useful when
   you have many features and want automatic selection.

**NodeGAM**
   Node-based generalized additive model. Efficient for large datasets.

Task Types
----------

Regression
~~~~~~~~~~

For predicting continuous values:

.. code-block:: python

   from nampy.models import NAMRegressor
   
   model = NAMRegressor()
   model.fit(X_train, y_train, max_epochs=100)
   predictions = model.predict(X_test)

Most models have a `*Regressor` variant (for example, `GPNAMRegressor`,
`IGANNRegressor`, `SIANRegressor`, `TreeNAMRegressor`, and `SNAMRegressor`).

Classification
~~~~~~~~~~~~~~

For predicting categorical labels:

.. code-block:: python

   from nampy.models import NAMClassifier
   
   model = NAMClassifier()
   model.fit(X_train, y_train, max_epochs=150)
   
   # Class predictions
   predictions = model.predict(X_test)
   
   # Probability estimates
   probabilities = model.predict_proba(X_test)

Most models have a `*Classifier` variant (e.g., `GPNAMClassifier`,
`SIANClassifier`, and `NBMClassifier`).

Distributional Regression (LSS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For modeling the full distribution of the target:

.. code-block:: python

   from nampy.models import NAMLSS
   
   model = NAMLSS(family="normal")
   model.fit(X_train, y_train, max_epochs=150)
   
   # Distribution parameters
   params = model.predict(X_test)

Available distribution families:

* ``normal`` - Normal/Gaussian distribution
* ``poisson`` - Poisson distribution (count data)
* ``gamma`` - Gamma distribution (positive continuous)
* ``beta`` - Beta distribution (values in [0, 1])
* ``studentt`` - Student's t-distribution (heavy tails)
* ``negativebinom`` - Negative binomial (overdispersed counts)
* ``inversegamma`` - Inverse gamma distribution
* ``categorical`` - Categorical distribution
* ``dirichlet`` - Dirichlet distribution
* ``quantile`` - Quantile regression
* ``robustnormal`` - Robust normal distribution

Most models have an `*LSS` variant (for example, `GPNAMLSS`, `SIANLSS`, and
`SNAMLSS`). `QNAM` is distributional-only.

Hyperparameter Configuration
-----------------------------

Common Hyperparameters
~~~~~~~~~~~~~~~~~~~~~~

All models share these common hyperparameters:

.. code-block:: python

   model = NAMRegressor(
       # Learning parameters
       lr=1e-4,                    # Learning rate
       lr_patience=10,             # Patience for LR scheduler
       lr_factor=0.1,              # LR reduction factor
       weight_decay=1e-6,          # L2 regularization
       
       # Architecture parameters
       layer_sizes=[128, 128, 32], # Hidden layer sizes
       dropout=0.5,                # Dropout rate
       
       # Preprocessing
       numerical_preprocessing="ple",  # Preprocessing strategy
       n_bins=50,                      # Number of bins for encoding
   )

Training Parameters
~~~~~~~~~~~~~~~~~~~

These are passed to the ``.fit()`` method:

.. code-block:: python

   model.fit(
       X_train, 
       y_train,
       max_epochs=150,      # Maximum training epochs
       lr=1e-4,             # Learning rate (can override init)
       patience=10,         # Early stopping patience
       val_size=0.2         # Validation split ratio
   )

Model Evaluation
----------------

Standard Metrics
~~~~~~~~~~~~~~~~

Use sklearn-compatible scoring:

.. code-block:: python

   # For regression
   r2 = model.score(X_test, y_test)
   
   # For classification
   accuracy = model.score(X_test, y_test)

Custom Metrics
~~~~~~~~~~~~~~

Use sklearn metrics manually:

.. code-block:: python

   from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
   
   predictions = model.predict(X_test)
   
   # Regression
   mse = mean_squared_error(y_test, predictions)
   
   # Classification
   acc = accuracy_score(y_test, predictions)
   f1 = f1_score(y_test, predictions, average='weighted')

Cross-Validation
~~~~~~~~~~~~~~~~

NAMpy models work with sklearn's cross-validation:

.. code-block:: python

   from sklearn.model_selection import cross_val_score
   
   model = NAMRegressor()
   scores = cross_val_score(
       model, X, y, cv=5,
       fit_params={'max_epochs': 50, 'lr': 1e-3}
   )
   
   print(f"CV Score: {scores.mean():.4f} (+/- {scores.std():.4f})")

Best Practices
--------------

Data Preparation
~~~~~~~~~~~~~~~~

1. **Handle missing values** before passing to NAMpy
2. **Encode categorical variables** appropriately
3. **Check for outliers** in your target variable
4. **Split data** into train/val/test sets

Training
~~~~~~~~

1. **Start with defaults** and adjust based on validation performance
2. **Use early stopping** (patience parameter) to prevent overfitting
3. **Monitor training** on a validation set
4. **Try different preprocessing** strategies for your data type

Model Selection
~~~~~~~~~~~~~~~

1. **Start with NAM** as a baseline
2. **Try specialized models** (GPNAM, NBM, etc.) for specific needs
3. **Compare multiple models** on validation data
4. **Consider interpretability vs. performance** trade-offs

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use GPU** if available (automatic with PyTorch)
2. **Batch your data** appropriately
3. **Adjust n_bins** based on dataset size
4. **Use appropriate preprocessing** for your feature types

Next Steps
----------

* Learn about :doc:`user_guide/preprocessing`
* Explore :doc:`user_guide/custom_models`
* Check :doc:`examples/index` for real-world use cases
* Read the :doc:`api/index` for detailed API information
