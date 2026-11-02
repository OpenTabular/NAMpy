How should neighboring regions share a spatial pattern?
=======================================================

A measurement taken in one region is often related to the measurements next to it. Independent regional intercepts ignore that adjacency. A Markov random field penalizes differences between neighbors that you declare in a graph.

The :doc:`regional-smoothing tutorial </notebooks/regional_smoothing>` simulates responses on a graph. It is not a pollution monitoring network. The comparison is an MRF smooth against independent region effects, scored on held-out rows from the same regions.

What is the problem?
--------------------

The question is whether the fitted regional pattern should be encouraged to vary smoothly across neighbors, and whether that changes predictions.

Nine regions, named ``R0`` through ``R8``, sit on a three-by-three grid. Two regions are neighbors when their grid steps differ by one in a single direction, so corners have two neighbors and the center has four. The penalty uses that graph. It does not treat the region names as numbers on a line. The true regional level is ``0.7 × column − 0.4 × row``. Regions ``R2`` and ``R6`` have 18 observations each. The other seven have 70. Every observation is that regional level plus standard normal noise, so the generating noise has standard deviation 1. The split is stratified by region, with ``test_size=0.25`` and ``random_state=109``. Held-out rows therefore come from regions that also appear in training.

Why is a linear model insufficient?
-----------------------------------

A single linear term has no notion of which regions share a border. Independent region effects are the more relevant baseline: ``y ~ region`` gives every region its own unpenalized coefficient and no penalty for disagreeing with a neighbor. They can overfit regions with few observations, which in this design are the two regions with 18 rows rather than 70.

Why this model?
---------------

The neighbor graph is part of the model specification. Corners have two neighbors, edge centers have three, and the middle region has four. The color of each point is the generating level ``0.7 × column − 0.4 × row``.

.. image:: /_static/applications_regions_graph.png
   :alt: Three-by-three region graph with an edge between regions one grid step apart and point color equal to the generating regional level
   :class: scientific-figure
   :align: center

The MRF smooth passes those neighbor lists through ``xt``:

.. code-block:: text

   y ~ s(region, bs="mrf", xt={"nb": neighbors})

``GAM`` fits that formula with REML. The independent-region fit is ``GAM(formula="y ~ region")`` on the same rows. Both are predicted on the same held-out rows. A second prediction, on a one-row-per-region grid in region order, is what the effect plot compares with the known levels ``0.7 × column − 0.4 × row``. That plot is where pooling is visible even when the RMSE values are close.

What the fit showed
-------------------

.. image:: /_static/applications_regions_effects.png
   :alt: Generating regional levels, independent region coefficients, and the Markov random field smooth on the nine regions
   :class: scientific-figure
   :align: center

The dashed line with points is the generating level of each region, in order from ``R0`` to ``R8``. The other two lines are the independent-region fit and the MRF fit, each predicted on a one-row-per-region grid. Test RMSE is 0.947 for independent regional effects and 0.946 for the MRF. The generating noise has standard deviation 1, so both RMSEs sit near the variation that a regional mean cannot remove. This realization supplies little evidence of a predictive gain. The plot is where the graph's pooling shows up: the MRF line stays closer to its neighbors, including the two regions that have 18 observations rather than 70.

Limits of the interpretation
----------------------------

A nearly identical RMSE does not mean the two fits assign the same level to every region. The graph is part of the model: a wrong neighbor list changes the penalty. Held-out rows from regions that also appear in training do not answer how the model would behave in a region with no data. A spherical smooth, for locations on a globe rather than a discrete graph, is a different tutorial: :doc:`Fit a smooth on the sphere </notebooks/spherical_smoothing>`.

:download:`Download notebook <../notebooks/regional_smoothing.ipynb>`
