Joint spline and neural composition
====================================

``SemiStructuredRegressor``, ``SemiStructuredClassifier`` and ``SemiStructuredLSS``
jointly optimize structured coefficients and a learned neural representation.
Scalar-response regression, binary/multiclass classification and scalar-response
distributional objectives reuse their existing neural losses and transformations.
This route uses the existing statistical formula compiler for a structured
block and the existing neural estimator lifecycle for preprocessing, validation,
component extraction and persistence.

.. literalinclude:: /../examples/example_semistructured.py
   :language: python

Inspect the fitted decomposition
--------------------------------

.. figure:: /_static/features/semistructured_components.png
   :alt: Structured contribution, projected neural remainder and their total along a new fixed-z predictor slice, compared with the simulated true mean.
   :width: 100%

   The held-out RMSE is 0.282. Along the illustrated z=0.8 slice, the
   neural contribution ranges from approximately -0.93 to 0.80. The total
   follows the main pattern but still misses part of the true mean near
   the boundaries.

The :doc:`complete tutorial </notebooks/semistructured_remainder>` shows the
subject split and provides optional decomposition checks.

Training-space identification
-----------------------------

Let B be the structured design, H the learned representation and W the training
case-weight matrix. Each full-cohort optimization step obtains a weighted QR
projection A of H onto B. The predictor is B beta + (H - B A) gamma. Gradients
flow through this projection, the representation and both coefficient blocks.
Thus B' W (H - B A) vanishes on training rows, including the intercept direction
when present. A linear representation entirely in B's span contributes zero.

Validation and prediction use the projection fitted on training rows. They
never refit a projection on a query cohort; predictions are independent of the
other rows included in a prediction call. Population identification, causal
interpretations and invariance to changing the structured specification do not
follow from this finite-training-design constraint.

``DefaultSemiStructuredConfig`` owns the RHS ``structured_formula``, selected
``neural_features``, ``hidden_dims``, ``representation_dim`` and default
``k``/``basis``. With no formula, all numeric inputs receive linear structured
effects and categorical inputs receive fitted factor contrasts. Formula names
refer to scalar preprocessed numerical features and one-hot categorical blocks; preprocessing and
basis construction are fitted on training data only.

``SemiStructuredTrainingConfig`` selects native CPU float64, full-cohort Adam
optimization. It owns ``epochs``, ``lr``, validation ``patience``,
``neural_weight_decay`` and fixed ``smoothing_params``. The inherited batch-size
control affects inference batching, not training minibatches. Fixed spline
roughness penalties and neural weight decay have separate parameter owners.
Early stopping selects the best joint validation state, including its projection.
``training__lr`` and other nested updates support sklearn parameter search;
updates invalidate fitted state. Completion means the configured optimization
run completed, not that a global optimum was certified.

Predictions and limits
----------------------

``predict_components`` reconstructs the additive predictor with structured
terms and an explicitly global ``neural_component``. The latter depends jointly
on its selected input features. LSS predictions use the existing
predictor-to-natural-parameter transformation; conditional distribution spread
is not coefficient uncertainty.

This route requires scalar numeric features or fitted one-hot categorical
blocks and a full-rank structured design under positive training weights.
Unknown categories raise explicitly. Row-specific offsets have exact predictor
width: (n, p), with a vector accepted only when p=1. Supply them to fit (and
offset_val for explicit validation) and again to prediction; offsets remain
separate from learned contributions. LSS offsets use raw predictor coordinates,
not natural-parameter coordinates. Scalar normal, gamma, beta, Student-t,
lognormal and Weibull composition have explicit fitted prediction checks.
Vector features/responses, formula offsets and Lightning gradient-training
configurations are unsupported. There is no joint REML, smoothing selection, neural coefficient
covariance, curve confidence band, survival composition, or multimodal encoder
support. Rank failures raise rather than dropping columns or introducing jitter.

The fitted artifact retains weights, compiled basis/schema, selected projection,
and fit diagnostics. It excludes the optimizer and raw training/validation
tensors. Failed refits preserve the previously published fitted model.

Validation covers weighted orthogonality, actual encoder updates, additive
reconstruction, new-cohort/batch invariance, persistence, sklearn cloning,
failed refits and explicit unsupported routes. See
:doc:`/neural_reference/semistructured` for parameter defaults.

References
----------

The structured/deep decomposition and orthogonalization principle are described
by [Ruegamer2024]_. NAMpy implements the fixed-penalty, full-cohort projection
route documented above. The paper's broader model/data-modality framework does
not expand those capabilities. See :doc:`/references` for the publication link.
