Training neural models
======================

Neural estimators have four configuration groups: ``architecture_config``,
``objective``, ``training`` and ``preprocessing``. Training controls belong to the
constructor. ``fit`` accepts features, targets, explicit validation data,
observation weights and known predictor offsets.

Configuration ownership
-----------------------

``architecture_config`` is the architecture's dataclass instance, such as
``DefaultNAMConfig``. It owns representation choices, including layer widths,
feature layers, dropout and interactions. Architectures read this record directly;
there is no flat override channel.

``ObjectiveConfig`` owns the loss or distribution family, distributional family
options, class weights and sampling strategy. Estimator roles determine regression,
classification or distributional target semantics.

``PreprocessingConfig`` contains PreTab feature transformation options. Architecture
preprocessing defaults apply to unspecified options; explicit options replace them.
The fitted preprocessing state is learned from training rows only.

``training`` selects a supported computational route. ``None`` resolves its default
from the architecture and actual objective, including the observed class count.

.. code-block:: python

    from nampy import NAMRegressor, GradientTrainingConfig, PreprocessingConfig
    from nampy.neural.configs import DefaultNAMConfig

    model = NAMRegressor(
        architecture_config=DefaultNAMConfig(layer_sizes=[32, 16], dropout=0.1),
        preprocessing=PreprocessingConfig(numerical_preprocessing="standardization"),
        training=GradientTrainingConfig(max_epochs=50, batch_size=128,
                                        random_state=7, lr=0.001),
    )
    model.fit(X, y, X_val=X_validation, y_val=y_validation,
              sample_weight=training_weights)

Weighted and imbalanced classification
---------------------------------------

Observation weights, class weights and resampling change different parts of
training. ``sample_weight`` weights individual rows in the loss. For classifiers,
``ObjectiveConfig(class_weight="balanced")`` derives inverse-frequency class
weights from the supplied training labels; a mapping can specify them explicitly.
When both are supplied, class weights multiply observation weights.

This small gradient-training example keeps validation rows and their weights
explicit. Validation selects the retained epoch; use a separate test cohort for
final assessment.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from nampy import NAMClassifier, ObjectiveConfig, GradientTrainingConfig
   from nampy.neural.configs import DefaultNAMConfig

   rng = np.random.default_rng(7)
   X = pd.DataFrame(rng.normal(size=(120, 2)), columns=["x", "z"])
   y = (X.x.to_numpy() + 0.2 * X.z.to_numpy() > 0.8).astype(int)
   weights = np.linspace(0.5, 1.5, len(X))
   model = NAMClassifier(
       architecture_config=DefaultNAMConfig(layer_sizes=[8]),
       objective=ObjectiveConfig(class_weight="balanced"),
       training=GradientTrainingConfig(max_epochs=5, batch_size=32, random_state=7),
   ).fit(
       X.iloc[:80], y[:80], sample_weight=weights[:80],
       X_val=X.iloc[80:], y_val=y[80:], sample_weight_val=weights[80:],
   )

``sample_weight_val`` controls validation loss and requires an explicit validation
cohort. Training class weights are not automatically applied to that cohort.
For balanced minibatch draws instead, use
``ObjectiveConfig(sampling_strategy="balanced")``: this samples training rows
with replacement using inverse class frequencies. It leaves validation rows
unchanged and requires ``GradientTrainingConfig``; native/fixed-design routes
reject resampling. Combining balanced sampling with class weights applies both
changes, so choose that combination deliberately. These class controls are not
available for regression, LSS or survival objectives. See
:doc:`/neural_reference/estimators/NAMClassifier` for the parameter contracts.

Gradient training
-----------------

``GradientTrainingConfig`` owns ``max_epochs``, ``max_steps``, ``max_time``, batching,
validation splitting, optimizer, scheduling and stopping controls. ``trainer_kwargs``
contains additional Lightning Trainer options, such as logging or accelerator
selection. Controls already owned by this record must be set on the record rather
than repeated in ``trainer_kwargs``.

Optimizer choices include ``adam``, ``adamw``, ``adagrad``, ``sgd`` and optional
``qhadam``. ``optimizer_kwargs`` supplies optimizer-specific arguments.
``lr`` and ``weight_decay`` configure the optimizer. ``lr_schedule`` selects
``plateau``, ``inverse_sqrt``, ``cosine``, ``warmup_cosine`` or ``none``;
``lr_patience``, ``lr_factor``, ``lr_warmup_steps``, ``lr_decay_steps`` and
``lr_decay_factor`` configure the selected schedule.

``monitor``, ``mode`` and ``patience`` control early stopping and best-checkpoint
selection. ``early_stopping_steps`` instead measures patience in optimizer steps.
``average_checkpoints`` and ``n_last_checkpoints`` select arithmetic averaging of
recent epoch-end weights. These are gradient-training operations.

Checkpoints use a temporary directory by default, removed after the selected
weights are loaded, including when training fails. Set ``checkpoint_path`` to
retain a separate checkpoint directory for each fit. ``best_model_path_`` is
``None`` when temporary checkpoints have been removed.

``feature_dropout`` drops complete term contributions independently per row,
with one mask shared across the term's output columns. Returned training terms
include that mask and reconstruct the prediction together with the intercept.
Evaluation disables this dropout. Hidden-layer dropout remains configured by
the architecture's separate controls.

Prediction preprocesses and evaluates batches of at most 1024 rows by default;
``batch_size`` overrides that limit. Results are accumulated on CPU, and the
model's previous training/evaluation state is restored after prediction.

Masked reconstruction uses ``pretrain_epochs``, ``pretrain_max_steps``,
``pretraining_ratio``, ``pretraining_noise`` and ``pretraining_feature_mask``.
Architectures without the required capability reject pretraining explicitly.
See the :ref:`NodeGAM pretraining recipe <nodegam-masked-pretraining>` for a
supported fit and the meaning of transferred state.

Fixed designs and native fitting
--------------------------------

``FixedDesignTrainingConfig`` selects the conjugate-gradient weighted ridge solve
for architectures exposing a fixed linear design, currently GPNAM with the default
regression loss. It owns ``ridge``, ``cg_rtol`` and ``cg_max_iter``. With no explicit
validation data, preprocessing and coefficient fitting use every supplied row.
Classification, LSS and custom losses do not support this route.

``NativeTrainingConfig`` defines common execution controls for an architecture-owned
algorithm. ``IGANNTrainingConfig`` adds IGANN boosting, ridge-initialization,
early-stopping and sparse-selection controls. It contains no unused gradient
optimizer or scheduler settings. IGANN defaults to native fitting for default-loss
regression and binary classification; multiclass and distributional objectives
use gradient training. Requesting an unsupported native route raises an error.

Cloning, refitting and artifacts
--------------------------------

Use nested parameters for sklearn searches, for example ``training__lr``,
``architecture_config__dropout`` and ``preprocessing__n_bins``. ``set_params``
replaces configuration records without mutating caller-owned records and clears
fitted state. ``clone`` creates an unfitted estimator.

A fit builds preprocessing, the objective and a trained module in staged state.
Only a successful fit publishes that state. Failed refits preserve the previous
prediction model and completion evidence. ``training_config_`` records the resolved
training configuration used by the successful fit.

Set ``GradientTrainingConfig(warm_start=True)`` before fitting to reuse weights on
subsequent fits. The first fit initializes fresh weights. Warm starts load weights
into a new staged module; optimizer state is not resumed. Incompatible architectures
or output widths fail explicitly. A checkpoint path may also supply initial weights.

The fitted model retains its prediction module, objective, fitted preprocessing
and lightweight diagnostics. It does not retain a Lightning Trainer, callbacks,
training batches or training observations. ``save_model`` and ``load_model`` use
the shared versioned NAMpy artifact format. Previous artifact versions are rejected
without migration or old-module remapping; load artifacts only from trusted sources.

For LSS models, completed neural training is distinct from statistical convergence.
Probability operations and moments are available only where the shared law declares
them. BCT/BCPE probability support does not imply a supported fitting route.

Joint structured/neural training
--------------------------------

``SemiStructuredTrainingConfig`` uses full-cohort CPU optimization with a
training-fitted weighted projection, fixed structured smoothing parameters and
validation early stopping. It has no Lightning optimizer/callback controls.
See :doc:`semistructured` for its supported objectives and interpretation.
