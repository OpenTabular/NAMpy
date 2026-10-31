IGANN
=====

When to use it
--------------

Use IGANN when a sparse linear starting model and staged nonlinear boosting
are preferable to end-to-end neural optimization.

Model structure
---------------

IGANN begins with a sparse linear model and then adds feature-wise nonlinear
updates from extreme learning machines. Their hidden weights are fixed random
draws; output weights are estimated during boosting. For iteration :math:`m`,

.. math::

   \eta_m(x) = \eta_{m-1}(x)
   + \nu \sum_j h_{jm}(x_j),

where :math:`\nu` is the boosting rate. Numeric features receive independent
random hidden units, while categorical features remain reference-coded and
linear in each update.

Fit and inspect
---------------

.. code-block:: python


   from nampy.neural.configs.igann_config import DefaultIGANNConfig
   from nampy.neural.configuration import IGANNTrainingConfig
   from nampy.models import IGANNRegressor

   model = IGANNRegressor(architecture_config=DefaultIGANNConfig(n_hid=10, n_estimators=500), training=IGANNTrainingConfig(boost_rate=0.1, early_stopping=30)).fit(X_train, y_train)

   prediction = model.predict(X_test)
   components = model.predict_components(X_test, center=True)
   figures = model.plot_terms(X_test, center=True)
   history = model.training_history()
   basis_info = model.basis_metadata()
   complexity = model.model_complexity()

``history`` records the native boosting loss path, rather than Lightning epochs.
``basis_info`` describes the retained hidden-feature basis; ``complexity`` reports
parameter counts and native quantities such as the fitted boosting-stage count.
These hooks describe the fitted representation and optimization, not inferential
coverage. See :doc:`training_history </neural_reference/methods/training_history>`
and :doc:`model_complexity </neural_reference/methods/model_complexity>` for their
outputs and architecture eligibility. The loss-path entries are populated by
native boosting; they do not log generic gradient training.

Interpretation and limits
-------------------------

IGANN owns its staged boosting procedure instead of using the common
end-to-end neural training loop. The fitted estimator records the number of
boosting stages, selected features, and training history.

Available estimators
--------------------

* :class:`nampy.models.IGANNRegressor`
* :class:`nampy.models.IGANNClassifier`
* :class:`nampy.models.IGANNLSS`

Reference
---------

Kraus et al. (2024), `Interpretable generalized additive neural networks
<https://doi.org/10.1016/j.ejor.2023.06.032>`_.

