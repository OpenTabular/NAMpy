Preprocessing
=============

Data preprocessing utilities.

.. currentmodule:: nampy.preprocessing

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Preprocessor

Overview
--------

The Preprocessor class handles all data preprocessing for NAMpy models,
including numerical feature encoding, categorical feature handling, and
various transformation strategies.

**Numerical Preprocessing Strategies:**

* ``standardization`` - Standardize features (mean=0, std=1)
* ``normalization`` - Normalize features to [0, 1]
* ``ple`` - Piecewise Linear Encoding
* ``binning`` - Bin numerical features
* ``one_hot`` - One-hot encode binned features

**Example:**

.. code-block:: python

   from nampy.preprocessing import Preprocessor
   
   preprocessor = Preprocessor(
       numerical_preprocessing="ple",
       n_bins=50,
       binning_strategy="quantile"
   )
   
   X_transformed = preprocessor.fit_transform(X, y)
   X_test_transformed = preprocessor.transform(X_test)
