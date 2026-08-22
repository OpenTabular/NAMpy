Quick Start Guide
=================

This guide provides a concise overview for getting started with NAMpy.

Basic Workflow
--------------

NAMpy follows the scikit-learn API and integrates with existing machine learning pipelines.

1. **Import** a model
2. **Initialize** with desired configuration
3. **Fit** on your training data
4. **Predict** on new data
5. **Evaluate** performance

Regression Example
------------------

Here's a complete example of using NAMpy for regression:

.. code-block:: python

   from nampy.models import NAMRegressor
   from sklearn.datasets import make_regression
   from sklearn.model_selection import train_test_split

   # Generate sample data
   X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Initialize model
   model = NAMRegressor(
       numerical_preprocessing="standardization",
       n_bins=50
   )

   # Train model
   model.fit(X_train, y_train, max_epochs=100, lr=1e-3)

   # Make predictions
   predictions = model.predict(X_test)

   # Evaluate
   r2_score = model.score(X_test, y_test)
   print(f"R² Score: {r2_score:.4f}")

Classification Example
----------------------

Classification works similarly:

.. code-block:: python

   from nampy.models import NAMClassifier
   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split

   # Generate sample data
   X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                               n_redundant=2, random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Initialize model
   model = NAMClassifier(
       numerical_preprocessing="ple",
       n_bins=50
   )

   # Train model
   model.fit(X_train, y_train, max_epochs=150, lr=1e-4)

   # Make predictions
   predictions = model.predict(X_test)
   probabilities = model.predict_proba(X_test)

   # Evaluate
   accuracy = model.score(X_test, y_test)
   print(f"Accuracy: {accuracy:.4f}")

Distributional Regression (LSS)
--------------------------------

NAMpy also supports distributional regression for modeling the full distribution:

.. code-block:: python

   from nampy.models import NAMLSS
   from sklearn.datasets import make_regression
   from sklearn.model_selection import train_test_split

   # Generate sample data
   X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Initialize model
   model = NAMLSS(family="normal")

   # Train model with distribution family
   model.fit(
       X_train, 
       y_train, 
       max_epochs=150, 
       lr=1e-4,
       patience=10,
   )

   # Predict distribution parameters
   dist_params = model.predict(X_test)

Working with DataFrames
-----------------------

NAMpy works with pandas DataFrames:

.. code-block:: python

   import pandas as pd
   from nampy.models import NAMRegressor

   # Load data
   df = pd.read_csv("data.csv")
   X = df.drop("target", axis=1)
   y = df["target"]

   # Train model
   model = NAMRegressor()
   model.fit(X, y, max_epochs=100)

   # Predict on new data
   new_data = pd.read_csv("new_data.csv")
   predictions = model.predict(new_data)

Preprocessing Options
---------------------

NAMpy provides several preprocessing strategies:

.. code-block:: python

   from nampy.models import NAMRegressor

   # Standardization (default for most models)
   model = NAMRegressor(numerical_preprocessing="standardization")

   # Min-max scaling
   model = NAMRegressor(numerical_preprocessing="minmax")

   # Piecewise Linear Encoding (PLE)
   model = NAMRegressor(numerical_preprocessing="ple", n_bins=50)

   # Custom bin expansion
   model = NAMRegressor(
       numerical_preprocessing="custombin",
       n_bins=50,
   )

   # One-hot encode categorical inputs
   model = NAMRegressor(categorical_preprocessing="one-hot")

Model Configurations
--------------------

Each model has a configuration class with default hyperparameters:

.. code-block:: python

   from nampy.models import NAMRegressor

   # Using default configuration
   model = NAMRegressor()

   # Customizing hyperparameters
   model = NAMRegressor(
       lr=1e-3,
       weight_decay=1e-5,
       dropout=0.3,
       layer_sizes=[128, 64, 32]
   )


   # Neural Attentive Tabular Transformer
   from nampy.models import NATTRegressor
   model = NATTRegressor()

   # Transformer-based NAM
   from nampy.models import NAMformerRegressor
   model = NAMformerRegressor()

Integration with scikit-learn
------------------------------

NAMpy models are fully compatible with scikit-learn utilities:

.. code-block:: python

   from nampy.models import NAMRegressor
   from sklearn.model_selection import cross_val_score, GridSearchCV

   model = NAMRegressor()

   # Cross-validation
   scores = cross_val_score(model, X, y, cv=5, 
                            fit_params={'max_epochs': 50, 'lr': 1e-3})
   print(f"CV Scores: {scores.mean():.4f} (+/- {scores.std():.4f})")

   # Hyperparameter tuning (note: this can be slow with deep learning)
   param_grid = {
       'dropout': [0.1, 0.3, 0.5],
       'n_bins': [25, 50, 100]
   }
   
   # Note: fit_params need to be passed separately for deep learning models
   # GridSearchCV with deep learning requires careful setup

Saving and Loading Models
--------------------------

Save trained models for later use:

.. code-block:: python

   import pickle
   from nampy.models import NAMRegressor

   # Train model
   model = NAMRegressor()
   model.fit(X_train, y_train, max_epochs=100)

   # Save model
   with open("nam_model.pkl", "wb") as f:
       pickle.dump(model, f)

   # Load model
   with open("nam_model.pkl", "rb") as f:
       loaded_model = pickle.load(f)

   # Use loaded model
   predictions = loaded_model.predict(X_test)
