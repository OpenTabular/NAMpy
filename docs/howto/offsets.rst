Predict with an Offset
======================

An offset is a known contribution to the linear predictor. Supply it during
fitting and again for new rows when it cannot be recovered from the formula:

.. code-block:: python

   model = GAM(
       formula="count ~ s(age)",
       family="poisson",
       optimize_smoothing=True,
       smoothing_method="reml",
   ).fit(data, offset=np.log(data["exposure"]))

   prediction = model.predict(
       new_data,
       type="response",
       offset=np.log(new_data["exposure"]),
   )

``predict_components`` records the offset separately so the reported terms
still reconstruct the link-scale prediction.
