Fit a regression, classification, or distributional model
=========================================================

These examples show the minimum fitting and prediction calls for three common tasks. Each assumes that training and test data have already been separated. The linked tutorials show how to prepare the data and assess the fitted models.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from sklearn.datasets import make_regression, make_classification
   from sklearn.model_selection import train_test_split
   from nampy.models import NAMRegressor, NAMClassifier, NAMLSS

Predict a continuous response
-----------------------------

.. code-block:: python


   from dataclasses import replace
   from nampy.neural.configuration import GradientTrainingConfig
   from nampy.neural.configuration import PreprocessingConfig


   X, y = make_regression(
       n_samples=400,
       n_features=6,
       n_targets=1,
       noise=0.2,
       random_state=7,
   )
   X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=7
   )

   model = NAMRegressor(preprocessing=PreprocessingConfig(numerical_preprocessing='standardization'))
   model.set_params(training=replace(model.training or GradientTrainingConfig(), max_epochs=20, batch_size=64, random_state=7)).fit(X_train, y_train)
   predictions = model.predict(X_test)
   regression_preview = pd.DataFrame({"Prediction": predictions[:3]})
   effects = model.plot_terms(X_test, center=True, reference_X=X_train)

.. include:: /_generated/howto_recipe_regression.inc

``predictions`` contains one response per test row. ``effects`` contains
feature-contribution figures on the additive-predictor scale.


.. code-block:: python

   regression_train, regression_test = X_train.copy(), X_test.copy()
   regression_target = y_train.copy()

Predict class probabilities
---------------------------

.. code-block:: python


   from dataclasses import replace
   from nampy.neural.configuration import GradientTrainingConfig
   from nampy.neural.configuration import PreprocessingConfig


   X, y = make_classification(
       n_samples=400,
       n_features=8,
       n_informative=5,
       random_state=11,
   )
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=11, stratify=y
   )

   model = NAMClassifier(preprocessing=PreprocessingConfig(numerical_preprocessing='standardization'))
   model.set_params(training=replace(model.training or GradientTrainingConfig(), max_epochs=20, batch_size=64, random_state=7)).fit(X_train, y_train)

   labels = model.predict(X_test)
   probabilities = model.predict_proba(X_test)
   classification_preview = pd.DataFrame({"Label": labels[:3],
       **{f"P(class={label})": probabilities[:3, j]
          for j, label in enumerate(model.classes_)}})

.. include:: /_generated/howto_recipe_classification.inc

``labels`` contains the selected class for each row. ``probabilities`` has one
column per class and is the appropriate output for threshold selection,
calibration, and probability-based scoring.

Predict a response distribution
-------------------------------

.. code-block:: python

   from dataclasses import replace
   from nampy.neural.configuration import GradientTrainingConfig
   from nampy.neural.configuration import ObjectiveConfig
   from nampy.neural.configuration import PreprocessingConfig


   X, y = make_regression(
       n_samples=400, n_features=6, noise=2.0, random_state=13
   )
   X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
   X_train, X_test, y_train, _ = train_test_split(
       X, y, test_size=0.2, random_state=13
   )

   model = NAMLSS(objective=ObjectiveConfig(family='normal'), preprocessing=PreprocessingConfig(numerical_preprocessing='standardization'))
   model.set_params(training=replace(model.training or GradientTrainingConfig(), max_epochs=20, batch_size=64, random_state=7)).fit(X_train, y_train)
   distribution_parameters = model.predict_parameters(X_test)
   distribution_preview = pd.DataFrame(distribution_parameters[:3],
                                        columns=["mean", "scale"])
   figures = model.plot_distributional_effects(
       X_test,
       reference_X=X_train,
       feature_names=["feature_0"],
   )


.. include:: /_generated/howto_recipe_distribution.inc

For a normal family, the two prediction columns are conditional location and
standard deviation. The plotted black envelope represents conditional spread,
not uncertainty about the fitted location effect.

Inspect multiple outputs
------------------------

A two-dimensional target produces one prediction column per output. Inspect
its component arrays directly: the current ``plot_terms`` renderer skips
multi-column contributions.

.. code-block:: python


   from dataclasses import replace
   from nampy.neural.configuration import GradientTrainingConfig
   from nampy.neural.configuration import PreprocessingConfig

   y_multi = np.column_stack([regression_target, 0.5 * regression_target])
   multi_model = NAMRegressor(preprocessing=PreprocessingConfig(numerical_preprocessing='standardization'))
   multi_model.set_params(training=replace(multi_model.training or GradientTrainingConfig(), max_epochs=20, batch_size=64, random_state=7)).fit(regression_train, y_multi)
   multi_predictions = multi_model.predict(regression_test)
   components = multi_model.predict_components(regression_test)
   for name, contribution in components.terms.items():
       print(name, np.asarray(contribution).shape)  # test rows, two outputs
