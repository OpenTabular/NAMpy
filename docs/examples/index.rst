Examples
========

This section provides practical examples of using nampy for various tasks.

.. toctree::
   :maxdepth: 2

   basic_regression
   basic_classification
   distributional_regression
   custom_model

Jupyter Notebook Examples
-------------------------

Interactive Jupyter notebooks are available with step-by-step tutorials:

.. toctree::
   :maxdepth: 1
   :caption: Interactive Tutorials

   ../notebooks/01_nam_regression
   ../notebooks/02_classification
   ../notebooks/03_distributional_regression
   ../notebooks/04_model_comparison
   ../notebooks/05_interpretability

Quick Examples
--------------

Basic Regression
~~~~~~~~~~~~~~~~

Complete regression workflow:

.. code-block:: python

   from nampy.models import NAMRegressor
   from sklearn.datasets import make_regression
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import mean_squared_error, r2_score
   
   # Generate data
   X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   # Train model
   model = NAMRegressor(numerical_preprocessing="standardization")
   model.fit(X_train, y_train, max_epochs=100, lr=1e-3)
   
   # Evaluate
   predictions = model.predict(X_test)
   print(f"MSE: {mean_squared_error(y_test, predictions):.4f}")
   print(f"R²: {r2_score(y_test, predictions):.4f}")

Basic Classification
~~~~~~~~~~~~~~~~~~~~

Binary and multi-class classification:

.. code-block:: python

   from nampy.models import NAMClassifier
   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score, classification_report
   
   # Generate data
   X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                               n_classes=3, random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   # Train model
   model = NAMClassifier(numerical_preprocessing="ple", n_bins=50)
   model.fit(X_train, y_train, max_epochs=150, lr=1e-4)
   
   # Evaluate
   predictions = model.predict(X_test)
   probabilities = model.predict_proba(X_test)
   
   print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
   print("\nClassification Report:")
   print(classification_report(y_test, predictions))

Distributional Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~

Modeling the full distribution:

.. code-block:: python

   from nampy.models import NAMLSS
   from sklearn.datasets import make_regression
   from sklearn.model_selection import train_test_split
   
   # Generate data
   X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   # Train model
   model = NAMLSS()
   model.fit(X_train, y_train, max_epochs=150, lr=1e-4, family="normal")
   
   # Predict distribution parameters
   params = model.predict(X_test)
   print(f"Predicted distribution parameters shape: {params.shape}")

Real-World Example: House Price Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from nampy.models import NAMRegressor
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import mean_absolute_error, r2_score
   
   # Load data (example)
   # df = pd.read_csv("house_prices.csv")
   
   # Separate features and target
   # X = df.drop("price", axis=1)
   # y = df["price"]
   
   # Split data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   # Train model
   model = NAMRegressor(
       numerical_preprocessing="ple",
       n_bins=50,
       layer_sizes=[128, 128, 32],
       dropout=0.3
   )
   
   model.fit(X_train, y_train, max_epochs=100, lr=1e-3, patience=10)
   
   # Evaluate
   predictions = model.predict(X_test)
   print(f"MAE: ${mean_absolute_error(y_test, predictions):,.2f}")
   print(f"R²: {r2_score(y_test, predictions):.4f}")

Model Comparison
~~~~~~~~~~~~~~~~

Comparing multiple models:

.. code-block:: python

   from nampy.models import NAMRegressor, GPNAMRegressor, NBMRegressor
   from sklearn.datasets import make_regression
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import r2_score
   
   # Generate data
   X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   models = {
       'NAM': NAMRegressor(),
       'GPNAM': GPNAMRegressor(),
       'NBM': NBMRegressor()
   }
   
   results = {}
   for name, model in models.items():
       model.fit(X_train, y_train, max_epochs=50, lr=1e-3)
       predictions = model.predict(X_test)
       score = r2_score(y_test, predictions)
       results[name] = score
       print(f"{name}: R² = {score:.4f}")
   
   # Best model
   best_model = max(results, key=results.get)
   print(f"\nBest model: {best_model}")

Hyperparameter Tuning
~~~~~~~~~~~~~~~~~~~~~

Using scikit-learn's GridSearchCV:

.. code-block:: python

   from nampy.models import NAMRegressor
   from sklearn.model_selection import GridSearchCV
   from sklearn.datasets import make_regression
   
   X, y = make_regression(n_samples=500, n_features=10, random_state=42)
   
   # Note: This can be slow with deep learning models
   model = NAMRegressor()
   
   param_grid = {
       'dropout': [0.1, 0.3, 0.5],
       'n_bins': [25, 50]
   }
   
   # GridSearchCV requires careful setup for deep learning
   # Consider using simpler validation instead
   
   # Better approach: Manual validation
   best_score = 0
   best_params = None
   
   for dropout in [0.1, 0.3, 0.5]:
       for n_bins in [25, 50]:
           model = NAMRegressor(dropout=dropout, n_bins=n_bins)
           model.fit(X_train, y_train, max_epochs=50, lr=1e-3)
           score = model.score(X_val, y_val)
           
           if score > best_score:
               best_score = score
               best_params = {'dropout': dropout, 'n_bins': n_bins}
   
   print(f"Best parameters: {best_params}")
   print(f"Best score: {best_score:.4f}")

Custom Model Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :doc:`custom_model` for a complete guide on implementing custom models.

More Examples
-------------

For more detailed examples, check out:

* :doc:`basic_regression` - Detailed regression examples
* :doc:`basic_classification` - Detailed classification examples
* :doc:`distributional_regression` - Distributional regression examples
* :doc:`custom_model` - Building custom models

Running Notebooks Locally
-------------------------

To run the example notebooks locally:

1. Clone the repository and navigate to the examples directory
2. Activate your conda environment: ``conda activate nampy``
3. Launch Jupyter: ``jupyter notebook``

The notebooks cover:

* **01_basic_regression** - Complete regression workflow with NAMRegressor
* **02_classification** - Binary and multi-class classification examples
* **03_distributional_regression** - Modeling uncertainty with NAMLSS
* **04_model_comparison** - Comparing different NAMpy model architectures
* **05_interpretability** - Visualizing and understanding model predictions
