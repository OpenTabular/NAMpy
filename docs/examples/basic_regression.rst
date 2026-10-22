Fit a neural regressor
======================

This recipe predicts synthetic electricity demand in megawatts from outdoor
temperature in degrees Celsius and hour on a 24-hour clock. The complete
:doc:`neural tutorial <../auto_examples/neural/plot_first_nam>` explains the
data-generating process and interprets the resulting curves; this page keeps
only the fit-and-predict pattern.

.. code-block:: python

   from sklearn.model_selection import train_test_split
   from nampy.models import NAMRegressor

   predictors = energy[["temperature", "hour"]]
   target = energy["demand"]
   X_train, X_test, y_train, y_test = train_test_split(
       predictors, target, test_size=0.2, random_state=21
   )

   model = NAMRegressor(numerical_preprocessing="minmax")
   model.fit(X_train, y_train, max_epochs=40, batch_size=128)
   predictions_mw = model.predict(X_test)
   figures = model.plot_terms(X_test, center=True, reference_X=X_train)

``predictions_mw`` contains one demand prediction per held-out observation.
The effect curves are contributions on the additive-predictor scale.
