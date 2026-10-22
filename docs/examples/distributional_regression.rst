Fit a distributional model
==========================

Suppose ``trips`` records commute distance in kilometres, whether a trip took
place during peak hours, and travel time in minutes. A distributional model can
learn that longer commutes have both a larger expected duration and greater
day-to-day variability.

.. code-block:: python

   from sklearn.model_selection import train_test_split
   from nampy.models import TreeNAMLSS

   X = trips[["distance_km", "peak_hour"]]
   y = trips["travel_time_min"]
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=29
   )

   model = TreeNAMLSS(family="normal", numerical_preprocessing="minmax")
   model.fit(X_train, y_train, max_epochs=40, batch_size=64)
   parameters = model.predict(X_test)
   figures = model.plot_distributional_effects(
       X_test,
       reference_X=X_train,
       feature_names=["distance_km"],
   )

For a normal family, the prediction columns are conditional location and
standard deviation. The plotted envelope represents conditional trip-to-trip
spread, not uncertainty about the fitted location effect. See the
:doc:`complete tutorial
<../auto_examples/distributional/plot_changing_variability>` for generated
data, output, and interpretation.
