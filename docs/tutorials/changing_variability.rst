Model a changing response distribution
======================================

Predicting the mean does not tell us how widely individual responses vary. In this example, the response has a curved mean and a standard deviation that increases with exposure. We fit a normal neural distributional model to learn both.

.. code-block:: python

   import numpy as np
   import pandas as pd

   rng = np.random.default_rng(29)
   exposure = rng.uniform(0.0, 1.0, 800)
   seasonal_index = rng.uniform(-1.0, 1.0, 800)
   X = pd.DataFrame({"exposure": exposure, "seasonal_index": seasonal_index})
   location = 1.2 * np.sin(2.0 * np.pi * exposure) + 0.35 * seasonal_index
   spread = 0.18 + 0.7 * exposure
   y = location + spread * rng.normal(size=800)


   from nampy.models import NAMLSS

   model = NAMLSS(
       family="normal",
       layer_sizes=[24, 12],
       dropout=0.0,
       numerical_preprocessing="minmax",
   ).fit(X, y, max_epochs=30, lr=1e-2, random_state=29)


NAMLSS predicts the parameters of the chosen response distribution. With the normal family, these are its mean and standard deviation. The effect plot shows how the fitted point prediction changes with exposure and how the conditional spread changes alongside it.

.. code-block:: python

   figures = model.plot_distributional_effects(
       X,
       reference_X=X,
       feature_names=["exposure"],
       point_label="conditional location effect",
       envelope_label="location effect ± conditional SD",
   )


.. image:: /_static/changing_variability.png
   :alt: Centered point effect with conditional response spread across exposure
   :class: scientific-figure

The blue curve is the centered point effect. The black curves are that effect plus and minus one conditional standard deviation, so their separation shows the fitted response spread. They are not confidence bounds for the blue curve. Red shading shows where input observations are concentrated.

The centering reference is the training data supplied through reference_X. The vertical level is therefore relative to that reference, rather than the full uncentered mean prediction.
