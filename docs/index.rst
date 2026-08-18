NAMpy: Interpretable (Additive) Tabular Deep Learning
=====================================================

.. image:: https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.11 or 3.12

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

NAMpy provides interpretable additive neural models for tabular data, with support for **regression**, **classification**, and **distributional regression** tasks.

Key Features
------------

* **Scikit-learn Compatible**: Consistent API with sklearn estimators
* **10+ Model Architectures**: NAM, GPNAM, NBM, NATT, NAMformer, and more
* **Three Task Types**: Regression, classification, and distributional regression (LSS)
* **Interpretable**: Additive structure supports feature-level interpretation
* **PyTorch Backend**: Built on modern deep learning tooling
* **Extensible**: Interfaces for custom model implementations

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install nampy

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   from nampy.models import NAMRegressor
   from sklearn.datasets import make_regression
   from sklearn.model_selection import train_test_split

   # Generate sample data
   X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

   # Train model
   model = NAMRegressor(numerical_preprocessing="standardization")
   model.fit(X_train, y_train, max_epochs=100, lr=1e-3)

   # Evaluate
   score = model.score(X_test, y_test)
   print(f"R² Score: {score:.4f}")

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide
   api/index
   models/index
   examples/index
   contributing
   changelog

.. toctree::
   :maxdepth: 1
   :caption: Additional Information:

   license
   faq

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
