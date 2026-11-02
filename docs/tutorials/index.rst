Tutorials
=========

The tutorials are complete, executed analyses. They are organized here as three tracks so introductory additive models, statistical GAM methods, and neural architectures are not one mixed list.

Pick the track that matches the question you arrived with. Later tracks assume the earlier ideas only where a page says so. The :doc:`worked questions </applications/index>` read one of these fits as a modeling question.

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: New to additive models
      :link: track_new
      :link-type: doc
      :img-top: ../_static/track_intro.png
      :img-alt: Points from a sine-shaped mean, the least-squares line through those points, and the mean itself
      :class-img-top: track-card-figure

      For Python users who know scikit-learn and have not fitted a GAM. Learn what an additive model is, read feature curves, diagnose a poor fit, and evaluate unseen rows.

   .. grid-item-card:: Statistical modeling with GAMs
      :link: track_statistical
      :link-type: doc
      :img-top: ../_static/track_statistical.png
      :img-alt: Equirectangular map of a latitude-longitude surface, the display used for a spherical smooth
      :class-img-top: track-card-figure

      For statisticians, R users, and scientific researchers. Spline bases, penalties, REML, model selection, uncertainty, shape constraints, GAMLSS, and diagnostics.

   .. grid-item-card:: Interpretable deep learning
      :link: track_neural
      :link-type: doc
      :img-top: ../_static/track_neural.png
      :img-alt: Interaction heatmap of two features, with three slices of that surface at fixed values of the second feature
      :class-img-top: track-card-figure

      For machine-learning researchers and model developers. NAM, NBM, NodeGAM, SIAN, interaction selection, distributional neural regression, and custom architectures.

If you already have an mgcv formula, start with :doc:`Coming from R's mgcv </from_mgcv>` and then use the statistical track for the surrounding workflow.

Seeded comparison scripts are in the `examples directory <https://github.com/OpenTabular/NAMpy/tree/main/examples>`_ on GitHub.

.. toctree::
   :maxdepth: 2
   :hidden:

   New to additive models <track_new>
   Statistical modeling with GAMs <track_statistical>
   Interpretable deep learning <track_neural>
   Worked questions <../applications/index>
