Compare NAM and NBM Feature Functions
=====================================

How much does the learned representation change the effects when the data and
training protocol stay fixed? This example fits a NAM with independent
feature networks and an NBM with shared learned bases.

.. code-block:: python

   from nampy.models import NAMRegressor, NBMRegressor

   nam = NAMRegressor(
       layer_sizes=[24, 12], dropout=0.0,
       numerical_preprocessing="minmax",
   ).fit(X, y, max_epochs=30, lr=1e-2, random_state=17)

   nbm = NBMRegressor(
       layer_sizes=[16], num_bases=8, nary=[1],
       dropout=0.0, bases_dropout=0.0,
       numerical_preprocessing="minmax",
   ).fit(X, y, max_epochs=30, lr=1e-2, random_state=17)

   nam_effects = nam.explain_terms(X, center=True, max_bins=28)
   nbm_effects = nbm.explain_terms(X, center=True, max_bins=28)

.. image:: ../_static/nam_nbm_comparison.png
   :alt: Centered NAM and NBM effects on the same plotting scale
   :class: scientific-figure
   :align: center

Both estimators return centered link-scale contributions through the same
interface. NAM assigns a separate network to each feature. NBM learns reusable
basis functions and combines them for each term, so similar curves do not
imply identical parameterizations. See :doc:`../models/nbm` before
interpreting terms created after categorical expansion.
