Tutorials
=========

Each tutorial begins with a modeling question, fits a model with the public
NAMpy API, and explains the resulting output. The datasets are synthetic so
the relevant structure is known in advance.

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Fit and diagnose a GAM
      :link: first_gam
      :link-type: doc

      Estimate nonlinear effects, evaluate held-out predictions, and check the
      chosen basis dimensions.

      .. image:: ../_static/first_gam_effects.png
         :alt: Smooth temperature and cyclic hour effects

   .. grid-item-card:: Compare a GAM and a NAM
      :link: gam_nam
      :link-type: doc

      Fit statistical and neural additive models to the same four features,
      then compare their centered contributions through one result contract.

      .. image:: ../_static/gam_nam_all_features.png
         :alt: GAM and NAM effects for every feature

   .. grid-item-card:: Encode monotonicity
      :link: monotone_effect
      :link-type: doc

      Compare an unconstrained smooth with a scientifically justified
      monotone effect.

      .. image:: ../_static/shape_constraint_comparison.png
         :alt: Unconstrained and monotone smooths

   .. grid-item-card:: Model a periodic effect
      :link: periodic_effect
      :link-type: doc

      Use a cyclic basis when the two ends of a feature represent the same
      boundary.

      .. image:: ../_static/first_gam_effects.png
         :alt: Cyclic effect of hour on energy demand

   .. grid-item-card:: Estimate an interaction surface
      :link: interaction_surface
      :link-type: doc

      Represent a joint spatial pattern that separate main effects cannot
      recover.

      .. image:: ../_static/interaction_surface.png
         :alt: Tensor-product interaction surface

   .. grid-item-card:: Compare NAM and NBM
      :link: nam_nbm
      :link-type: doc

      Fit two neural representations to the same data and compare centered
      effects on one scale.

      .. image:: ../_static/nam_nbm_comparison.png
         :alt: NAM and NBM feature effects

   .. grid-item-card:: Model changing variability
      :link: changing_variability
      :link-type: doc

      Estimate conditional location and spread instead of assuming constant
      residual variation.

      .. image:: ../_static/changing_variability.png
         :alt: Distributional feature effect and conditional spread

.. toctree::
   :maxdepth: 1
   :hidden:

   first_gam
   monotone_effect
   periodic_effect
   interaction_surface
   gam_nam
   nam_nbm
   changing_variability
