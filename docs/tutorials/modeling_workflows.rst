Four modeling examples
======================

These four examples show the main ways to fit additive models in NAMpy: a statistical GAM, a distributional GAM, a neural additive model, and a neural distributional model. Each includes fitted outputs and plots you can reproduce with the accompanying code.

Statistical GAM
---------------

We fit a smooth of x and a group coefficient. The response was generated without a group effect, so the fitted group coefficient is small and the two group prediction curves nearly overlap.

.. include:: /_includes/model_showcases.inc
   :start-after: .. showcase-gam-start
   :end-before: .. showcase-gam-end

.. include:: /_generated/home_gam.inc
   :end-before: For the first displayed row

The summary describes the fitted model. Effective degrees of freedom measure the smooth's fitted flexibility; explained deviance describes its fit to the training responses.

For the first row of the component table, the intercept and two contributions add to the prediction: -0.0111 - 0.0016 - 0.0512 is approximately -0.0639. The input columns identify the observation; they are not additional contributions.

.. image:: /_static/home_gam.png
   :alt: GAM smooth effect and fitted response by group
   :class: scientific-figure

For numerical inputs, GAMRegressor can also construct main-effect smooths without a formula. Here it is fitted on alternating rows and predicts the remaining rows. The reported standard errors concern the fitted mean, rather than the spread of future observations.

.. include:: /_includes/model_showcases.inc
   :start-after: .. showcase-gam-adapter-start
   :end-before: .. showcase-gam-adapter-end

.. include:: /_generated/home_gam_adapter.inc

.. image:: /_static/home_gam_adapter.png
   :alt: Held-out numerical GAM predictions with mean standard errors
   :class: scientific-figure

Distributional GAM
------------------

This normal GAMLSS fits a mean and a standard deviation. The mean depends on x and group; the scale depends on x only. predict returns the transformed distribution parameters, while raw=True returns their underlying predictors.

.. include:: /_includes/model_showcases.inc
   :start-after: .. showcase-gamlss-start
   :end-before: .. showcase-gamlss-end

.. include:: /_generated/home_gamlss.inc

.. image:: /_static/home_gamlss.png
   :alt: GAMLSS conditional means and standard deviations
   :class: scientific-figure

Fitted conditional means and standard deviations. The group means differ, while the scale curves overlap because the scale formula does not include group.

Neural additive model
---------------------

The NAM example uses four synthetic housing-like features with known nonlinear effects. The plots compare the learned effects with the generating functions after centering them over the same input rows.

.. include:: /_includes/model_showcases.inc
   :start-after: .. showcase-nam-start
   :end-before: .. showcase-nam-end

.. image:: /_static/homepage_demo_light.png
   :alt: Fitted NAM feature effects and known centered truth
   :class: scientific-figure

The dashed curves show the known effects. Red shading shows feature density, making sparsely observed ranges visible. The component table gives the intercept and feature contributions for individual predictions; the importance plot summarizes their average absolute magnitude.

.. include:: /_generated/home_nam.inc

.. image:: /_static/home_nam_importance.png
   :alt: NAM mean absolute centered contribution importance
   :class: scientific-figure

Neural distributional model
---------------------------

The final example fits a normal NAMLSS to a response whose mean and standard deviation both vary with the inputs. The blue curve shows a centered point effect. The black curves add and subtract one conditional standard deviation at the reference setting of the other terms, showing response spread rather than confidence bounds for the effect.

.. include:: /_includes/model_showcases.inc
   :start-after: .. showcase-neural-lss-start
   :end-before: .. showcase-neural-lss-end

.. image:: /_static/readme_distributional_effects.png
   :alt: Neural normal point effects and conditional standard deviation envelopes
   :class: scientific-figure
