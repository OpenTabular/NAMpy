Prediction Outputs and Plotting
===============================

Choose the prediction method according to the quantity you need: a response prediction, class probability, distribution parameter, quantile, or additive contribution. The table below summarizes the returned values and their shapes.

``n`` denotes observations, ``c`` classes, and ``q`` requested quantiles. Parameter counts and transforms are family-specific.

.. list-table::
   :header-rows: 1
   :class: api-reference-table

   * - Estimator/task
     - Method and shape
     - Meaning and plotting route
   * - Statistical or neural scalar regression
     - ``predict(X)``: ``(n,)``
     - Conditional response prediction. Neural multi-output regression returns
       ``(n, outputs)``; inspect component columns explicitly.
   * - Statistical ``GAMClassifier`` or neural classification
     - ``predict(X)``: ``(n,)``; ``predict_proba(X)``: ``(n, c)``
     - Labels or probabilities in ``classes_`` order. Binary components add in
       log-odds. Multiclass probability prediction does not imply component support.
   * - Statistical GAMLSS / neural LSS
     - ``predict(X)``: natural family columns; ``predict(X, raw=True)``: raw predictors
     - Normal natural columns are mean and standard deviation. Raw predictors
       have family-specific transforms; statistical and neural raw scales differ.
   * - Statistical ``GAMLSS`` or neural LSS point prediction
     - ``predict_point(X)``: family-defined point output
     - Uses the fitted family's point functional. Use
       ``plot_distributional_effects`` for supported point/spread displays.
   * - Quantile family / QNAM
     - ``predict(X)``: ``(n, q)``
     - Conditional quantiles, without a complete density. ``evaluate`` reports
       ``pinball_loss``, summed over levels and averaged over rows.
       Neural ``predict_quantiles(X, probabilities)`` selects explicitly fitted
       levels as ``(n, q)`` without inventing a complete density.
   * - Additive components
     - ``predict_components(X)``: ``AdditivePrediction``
     - Intercept, terms and optional offset reconstruct ``link``. ``response``
       applies the objective's response map. A term can be main, interaction,
       contextual or global; reconstruction alone establishes no identifiability.

Inspect a few returned rows
---------------------------

This statistical example shows labels beside probabilities in ``classes_``
order, then Normal parameters beside the point prediction. For this family,
``predict_point`` equals the ``mu`` column; ``sigma`` is a standard deviation.

.. literalinclude:: ../scripts/generate_howto_assets.py
   :language: python
   :start-after: # PREDICTIONS START
   :end-before: # PREDICTIONS END
   :dedent: 4

.. include:: /_generated/howto_classification_outputs.inc

.. include:: /_generated/howto_distribution_outputs.inc

Choose the display by contribution type
---------------------------------------

* A single-column numerical main effect can use neural ``plot_terms``.
  Supply named DataFrame columns. See :doc:`../user_guide/interpretability`
  for centering and reference data.
* Multiple output columns require an explicit per-output plot, as in
  :doc:`../models/qnam`. The single-column renderer skips them.
* Categorical terms can be summarized by level on the predictor scale, as in
  :doc:`../notebooks/05_real_data_classification`.
* Explicit interactions can use ``plot_interactions`` or a labeled surface.
* A global or contextual contribution needs a dependence/perturbation display;
  it is not a single-feature marginal curve. See :doc:`../notebooks/03_neural_models`.

Statistical GAM ``plot`` includes its own effect and covariance conventions.
A confidence band concerns an estimated function; a conditional standard
deviation concerns outcomes; an ensemble-member standard deviation concerns
between-fit disagreement. Name the actual quantity in each caption.

Pass new-row predictor offsets explicitly to prediction and components; see :doc:`offsets`.

Calibrated future-observation intervals
---------------------------------------

Use :doc:`../user_guide/calibration` for split-conformal intervals based on a
separate calibration cohort. These carry future-observation uncertainty rather
than coefficient uncertainty or intervals for an estimated conditional quantile.
Calibration does not change the fitting algorithm or probability law.
