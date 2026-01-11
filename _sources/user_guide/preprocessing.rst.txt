Preprocessing
=============

NAMpy provides flexible preprocessing options for tabular data.

Numerical Feature Preprocessing
--------------------------------

NAMpy supports several strategies for numerical features:

Standardization
~~~~~~~~~~~~~~~

Scale features to have mean=0 and std=1:

.. code-block:: python

   from nampy.models import NAMRegressor
   
   model = NAMRegressor(numerical_preprocessing="standardization")

Best for: Most cases, especially when features have different scales.

Normalization
~~~~~~~~~~~~~

Scale features to [0, 1]:

.. code-block:: python

   model = NAMRegressor(numerical_preprocessing="normalization")

Best for: When you want bounded features.

Piecewise Linear Encoding (PLE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Encodes continuous features using piecewise linear functions:

.. code-block:: python

   model = NAMRegressor(
       numerical_preprocessing="ple",
       n_bins=50
   )

Best for: Capturing non-linear relationships with interpretable models.

Binning
~~~~~~~

Discretize continuous features into bins:

.. code-block:: python

   model = NAMRegressor(
       numerical_preprocessing="binning",
       n_bins=50,
       binning_strategy="quantile"  # or "uniform"
   )

Best for: Reducing feature complexity, handling outliers.

One-Hot Encoding
~~~~~~~~~~~~~~~~

Bin and then one-hot encode:

.. code-block:: python

   model = NAMRegressor(
       numerical_preprocessing="one_hot",
       n_bins=50
   )

Best for: When you want categorical-like treatment of continuous features.

Categorical Feature Handling
-----------------------------

Automatic Detection
~~~~~~~~~~~~~~~~~~~

NAMpy automatically detects categorical features:

.. code-block:: python

   model = NAMRegressor(
       cat_cutoff=0.03  # Treat as categorical if <3% unique values
   )

Manual Control
~~~~~~~~~~~~~~

Force all integers to be numerical:

.. code-block:: python

   model = NAMRegressor(
       treat_all_integers_as_numerical=True
   )

Decision Tree-Based Binning
----------------------------

Use decision trees to determine optimal bin edges:

.. code-block:: python

   model = NAMRegressor(
       use_decision_tree_bins=True,
       numerical_preprocessing="binning"
   )

This uses the target variable to find informative bin boundaries.

Custom Preprocessing
--------------------

You can also preprocess data manually before passing to NAMpy:

.. code-block:: python

   from sklearn.preprocessing import StandardScaler
   from nampy.models import NAMRegressor
   
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X_train)
   
   model = NAMRegressor()
   model.fit(X_scaled, y_train, max_epochs=100)

Best Practices
--------------

1. **Try PLE first** for interpretable models (NAM, GPNAM)
2. **Use standardization** for deep models (NATT, NAMformer)
3. **Adjust n_bins** based on dataset size (25-100 typically)
4. **Use quantile binning** for skewed distributions
5. **Validate** preprocessing choices on a validation set

Preprocessing Pipeline
----------------------

The preprocessing happens automatically in the `.fit()` method:

.. code-block:: python

   model = NAMRegressor(numerical_preprocessing="ple", n_bins=50)
   
   # Preprocessing is applied internally
   model.fit(X_train, y_train, max_epochs=100)
   
   # Same preprocessing is applied at prediction time
   predictions = model.predict(X_test)

The preprocessor state is saved with the model, ensuring consistent
transformations at prediction time.

Accessing the Preprocessor
---------------------------

Advanced users can access the underlying preprocessor:

.. code-block:: python

   # After fitting
   preprocessor = model.preprocessor
   
   # Transform new data
   X_transformed = preprocessor.transform(X_new)

For more details, see :class:`nampy.preprocessing.Preprocessor`.

