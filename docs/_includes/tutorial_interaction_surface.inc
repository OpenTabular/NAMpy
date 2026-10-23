.. _tutorial-interaction:

Add a joint spatial effect
--------------------------

Can separate latitude and longitude effects describe a localized spatial
pattern? Usually not: the effect of one coordinate changes with the other. A
tensor-product smooth represents their joint contribution.

.. code-block:: python

   import numpy as np
   import pandas as pd

   rng = np.random.default_rng(33)
   latitude = rng.uniform(-1.0, 1.0, 700)
   longitude = rng.uniform(-1.0, 1.0, 700)
   response = (
       2.5 * np.exp(-4.0 * ((latitude - 0.25) ** 2 + (longitude + 0.2) ** 2))
       - 1.6 * np.exp(-5.0 * ((latitude + 0.45) ** 2 + (longitude - 0.4) ** 2))
       + rng.normal(0.0, 0.16, 700)
   )
   spatial_data = pd.DataFrame({"response": response, "latitude": latitude, "longitude": longitude})


   from nampy.gam import GAM

   model = GAM(
       formula=(
           "response ~ te(latitude, longitude, "
           "bs=c('cr', 'cr'), k=c(8, 8))"
       ),
       family="gaussian",
       optimize_smoothing=True,
       smoothing_method="reml",
   ).fit(spatial_data)

   figures = model.plot(pages=1, scheme=2, rug=False)

.. image:: ../_static/interaction_surface.png
   :alt: Estimated latitude and longitude interaction surface
   :class: scientific-figure
   :align: center

The color and contours show one joint term, not two marginal curves. Its
reported contribution depends on both named variables. Centering preserves
prediction reconstruction, but does not make main and interaction effects
unique when overlapping terms are included.
