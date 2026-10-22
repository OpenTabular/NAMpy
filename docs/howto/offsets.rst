Predict with an offset
======================

Suppose ``incident_count`` is the number of safety incidents recorded at each
site, ``operating_hours`` is the time each site was observed, and
``day_of_year`` describes seasonality. An offset accounts for unequal exposure
without estimating a coefficient for it.

.. code-block:: python

   model = GAM(
       formula="incident_count ~ s(day_of_year, bs='cc')",
       family="poisson",
       optimize_smoothing=True,
       smoothing_method="reml",
   ).fit(sites, offset=np.log(sites["operating_hours"]))

   predicted_incidents = model.predict(
       new_sites,
       type="response",
       offset=np.log(new_sites["operating_hours"]),
   )

``predict_components`` records the known exposure offset separately so the
reported terms still reconstruct the link-scale prediction.
