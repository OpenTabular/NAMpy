Recover step-shaped effects with additive trees
===============================================

Some feature effects change at thresholds rather than varying smoothly. We will fit TreeNAM and EnsembleTreeNAM to a response with step changes in age and income and a shift for each region. Since the effects are known, we can compare both predictions and recovered shapes.

Create step-shaped effects
--------------------------

Age and income each contribute a piecewise-constant function. Region adds a categorical shift, and independent noise is added to their sum. We reserve separate rows for training, early stopping, and testing.

.. code-block:: python

   import tempfile
   import numpy as np
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from nampy.models import TreeNAMRegressor, EnsembleTreeNAMRegressor


.. literalinclude:: ../../examples/example_ensemble.py
   :language: python
   :start-after: def main():
   :end-before: # Models
   :dedent: 4

Fit a tree and a joint ensemble
-------------------------------

A soft tree replaces hard branches with smooth routing probabilities. Tree depth controls how finely the input can be partitioned, while the evaluation settings determine whether routing remains soft or becomes hard.

EnsembleTreeNAM averages several member contributions and trains them jointly. It is different from fitting separate bootstrap models: disagreement between its joint members is not automatically an uncertainty estimate.

.. literalinclude:: ../../examples/example_ensemble.py
   :language: python
   :start-after: # Models
   :end-before: # Predictions / evaluation
   :dedent: 4

.. code-block:: python

   for fitted in (single_model, ensemble_model):
       fitted.predict_components(X_test).validate_additive_reconstruction(atol=1e-5)

.. image:: /_static/detailed_tree_effects.png
   :alt: Tree and joint-ensemble predictions against the noiseless response
   :class: scientific-figure

Predictions from the single tree and joint ensemble against the known noiseless response. The diagonal represents exact recovery of that signal.

.. include:: /_generated/detailed_tree_metrics.inc

The ensemble predicts more accurately on this test set. Its noisy-response MAE is 0.237, compared with 0.314 for the single tree, and its R2 rises from 0.803 to 0.882. The noiseless-signal MAE excludes the added response noise and measures recovery of the simulated mean.

Compare the feature effects
---------------------------

We center the fitted and true effects over the same training rows before comparing them. The age and income panels show whether the models recover the step locations and heights. The region panel compares the categorical shifts.

.. image:: /_static/detailed_tree_contributions.png
   :alt: Known and fitted age, income, and region contributions
   :class: scientific-figure

Known effects and fitted contributions for age, income, and region. Rounded transitions and missed step heights show where the tree representation or fit differs from the generating functions.

:download:`Download the complete example <../../examples/example_ensemble.py>`
