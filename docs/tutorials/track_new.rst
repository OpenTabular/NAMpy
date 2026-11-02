New to additive models
======================

This track is for Python users who fit scikit-learn models and have not used generalized additive models. An additive model writes the predictor as an intercept plus one function per feature. Those functions may be nonlinear. The inverse link then maps the sum to the response scale.

Work through the four tutorials in order. The first one fits two representations to data whose curves are known, so a test score can be compared with the shape of the fitted effects.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Discover nonlinear effects and check whether you can trust them
      :link: ../notebooks/01_getting_started
      :link-type: doc

      Fit a GAM and a NAM to the same simulation, compare held-out predictions, and read centered feature contributions against the relationships that generated the data.

      :download:`Download notebook <../notebooks/01_getting_started.ipynb>`

   .. grid-item-card:: How does demand change with temperature and time of day?
      :link: first_gam
      :link-type: doc

      Fit cubic and cyclic smooths to synthetic energy demand, score the held-out rows, and inspect residuals and basis dimension.

      :notebook-download:`first_gam`

   .. grid-item-card:: Use GAM diagnostics to improve a model
      :link: ../notebooks/gam_diagnostics
      :link-type: doc

      See what changes when a basis is too small, an interaction is missing, or correlated inputs make a single effect unstable.

      :download:`Download notebook <../notebooks/gam_diagnostics.ipynb>`

   .. grid-item-card:: Classification with missing values and categorical features
      :link: ../notebooks/05_real_data_classification
      :link-type: doc

      Compare a NAM classifier with logistic regression on the scikit-learn breast-cancer measurements, including held-out log loss and ROC-AUC.

      :download:`Download notebook <../notebooks/05_real_data_classification.ipynb>`

What each step is for
---------------------

The first tutorial defines the additive predictor and shows that similar test scores can come from different fitted curves. The demand tutorial is the first applied GAM: temperature is not linear, and hour of day is cyclic. The diagnostic tutorial is where a poor fit becomes a specific revision. The classification tutorial is the held-out check on measured data, with logistic regression as the baseline and preprocessing fit on the training rows only.

Effects in the classification model are contributions to the log-odds. Read :ref:`howto-user-guide-interpretability` before treating a curve as a change in probability.

Where to go next
----------------

Continue with :doc:`statistical modeling <track_statistical>` for penalties, families, and uncertainty, or :doc:`interpretable deep learning <track_neural>` for neural additive architectures. The :doc:`energy </applications/energy_demand>` and :doc:`housing-style </applications/housing_effects>` application pages reuse the second tutorial and the GAM–NAM comparison.

.. toctree::
   :maxdepth: 1
   :hidden:

   Discover nonlinear effects and check whether you can trust them <../notebooks/01_getting_started>
   How does demand change with temperature and time of day? <first_gam>
   Use GAM diagnostics to improve a model <../notebooks/gam_diagnostics>
   Classification with missing values and categorical features <../notebooks/05_real_data_classification>
