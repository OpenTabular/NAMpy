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
   * - Neural classification
     - ``predict(X)``: ``(n,)``; ``predict_proba(X)``: ``(n, c)``
     - Labels or probabilities in ``classes_`` order. Binary components add in
       log-odds. Multiclass probability prediction does not imply component support.
   * - Statistical GAMLSS / neural LSS
     - ``predict(X)``: natural family columns; ``predict(X, raw=True)``: raw predictors
     - Normal natural columns are mean and standard deviation. Raw predictors
       have family-specific transforms; statistical and neural raw scales differ.
   * - Neural LSS point prediction
     - ``predict_point(X)``: family-defined point output
     - Uses the fitted family's point functional. Use
       ``plot_distributional_effects`` for supported point/spread displays.
   * - Quantile family / QNAM
     - ``predict(X)``: ``(n, q)``
     - Conditional quantiles, without a complete density. ``evaluate`` reports
       ``pinball_loss``, summed over levels and averaged over rows.
   * - Additive components
     - ``predict_components(X)``: ``AdditivePrediction``
     - Intercept, terms and optional offset reconstruct ``link``. ``response``
       applies the objective's response map. A term can be main, interaction,
       contextual or global; reconstruction alone establishes no identifiability.

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

Neural offsets require explicit inference composition; see :doc:`offsets`.
