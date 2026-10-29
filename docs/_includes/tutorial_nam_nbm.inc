.. _tutorial-nam-nbm:

Compare independent networks with shared bases
----------------------------------------------

How much does the learned representation change the effects when the data and
training protocol stay fixed? This example fits a NAM with independent
feature networks and an NBM with shared learned bases.

.. code-block:: python

   import numpy as np
   import pandas as pd

   rng = np.random.default_rng(17)
   temperature = rng.uniform(-2.0, 2.0, 700)
   humidity = rng.uniform(-2.0, 2.0, 700)
   X = pd.DataFrame({"temperature": temperature, "humidity": humidity})
   y = 1.7 * np.sin(1.5 * temperature) + 0.65 * humidity**2 + rng.normal(0.0, 0.18, 700)


   from nampy.models import NAMRegressor, NBMRegressor
   from nampy.neural.configs.nam_config import DefaultNAMConfig
   from nampy.neural.configs.nbm_config import DefaultNBMConfig
   from nampy.neural.configuration import GradientTrainingConfig, PreprocessingConfig

   training = GradientTrainingConfig(max_epochs=30, lr=1e-2, random_state=17)
   preprocessing = PreprocessingConfig(numerical_preprocessing="minmax")

   nam = NAMRegressor(
       architecture_config=DefaultNAMConfig(layer_sizes=[24, 12], dropout=0.0),
       preprocessing=preprocessing,
       training=training,
   ).fit(X, y)

   nbm = NBMRegressor(
       architecture_config=DefaultNBMConfig(layer_sizes=[16], num_bases=8, nary=[1],
           dropout=0.0, bases_dropout=0.0),
       preprocessing=preprocessing,
       training=training,
   ).fit(X, y)

   nam_effects = nam.explain_terms(X, center=True, max_bins=28)
   nbm_effects = nbm.explain_terms(X, center=True, max_bins=28)

.. code-block:: python

   import matplotlib.pyplot as plt

   figure, axes = plt.subplots(2, 2, figsize=(9, 6), sharey="col")
   for row, (name, effects) in enumerate([("NAM", nam_effects), ("NBM", nbm_effects)]):
       for column, feature in enumerate(X.columns):
           term = effects[effects["term"] == feature].sort_values("value")
           axes[row, column].plot(term["value"], term["contribution"])
           axes[row, column].set(title=f"{name}: {feature}", xlabel="Feature value")
           if column == 0:
               axes[row, column].set_ylabel("Centered contribution")
   figure.tight_layout()

.. image:: ../_static/nam_nbm_comparison.png
   :alt: Centered NAM and NBM effects on the same plotting scale
   :class: scientific-figure
   :align: center

Both estimators return centered link-scale contributions through the same
interface. NAM assigns a separate network to each feature. NBM learns reusable
basis functions and combines them for each term, so similar curves do not
imply identical parameterizations. See :doc:`../models/nbm` before
interpreting terms created after categorical expansion.
