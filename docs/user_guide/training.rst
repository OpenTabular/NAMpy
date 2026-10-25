Training Models
===============

This guide covers gradient-trained neural estimators. Statistical GAMs and
architecture-native solvers use their model-specific fitting controls.

Basic Training
--------------

Gradient-trained neural estimators share the following fit controls:

.. code-block:: python

   from nampy.models import NAMRegressor

   model = NAMRegressor()
   model.fit(X_train, y_train, max_epochs=100, lr=1e-3)

Training Parameters
-------------------

Common Training Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

The defaults below describe ``NAMRegressor`` and its shared ``fit`` defaults.
Other architecture configurations may override optimizer settings.

.. list-table::
   :header-rows: 1
   :widths: 20 50 15 15

   * - Parameter
     - Description
     - Default
     - Typical Range
   * - ``max_epochs``
     - Maximum number of training epochs
     - 100
     - 50-300
   * - ``lr``
     - Learning rate
     - 1e-4
     - 1e-5 to 1e-2
   * - ``patience``
     - Early stopping patience (epochs)
     - 15
     - 5-30
   * - ``val_size``
     - Fraction of data for validation
     - 0.2
     - 0.1-0.3

Early Stopping
--------------

Early stopping monitors a validation criterion and can reduce overfitting:

.. code-block:: python

   model.fit(
       X_train,
       y_train,
       max_epochs=200,
       patience=15,              # Stop if val loss doesn't improve for 15 epochs
       val_size=0.2
   )

It selects a checkpoint according to the monitored validation criterion and
can save training time. It does not establish an optimal stopping point or
guarantee generalization. Keep final test rows outside checkpoint selection.

Learning Rate Scheduling
-------------------------

nampy automatically adjusts the learning rate during training:

.. code-block:: python

   model = NAMRegressor(
       lr=1e-3,                  # Initial learning rate
       lr_patience=10,           # Reduce LR after 10 epochs w/o improvement
       lr_factor=0.1             # Multiply LR by 0.1 when reducing
   )

The learning rate is reduced when validation loss plateaus.

``lr_schedule`` accepts ``"plateau"``, ``"inverse_sqrt"``, ``"cosine"``,
``"warmup_cosine"``, and ``"none"``. Step-wise schedules are selected at fit
time. ``cosine`` reproduces the schedule used by the NBM-SPAM reference trainer; adding warm-up steps yields
its warm-up/cosine variant.

.. code-block:: python

   model.fit(
       X_train,
       y_train,
       lr_schedule="cosine",
       lr_warmup_steps=100,
       lr_decay_steps=5000,  # cosine horizon; inferred when omitted
   )

Monitoring Training
-------------------

Training Progress
~~~~~~~~~~~~~~~~~

nampy uses PyTorch Lightning for training, which provides progress bars
and logging out of the box.

Custom Monitoring
~~~~~~~~~~~~~~~~~

For more control, pass Lightning trainer callbacks and loggers via
``**trainer_kwargs`` in ``fit()``.

GPU Training
------------

GPU benefit depends on network size, batch size, data transfer, and workload.
Small tabular models can be faster on CPU. For Lightning gradient fitting,
choose a device explicitly:

.. code-block:: python

   model.fit(X_train, y_train, accelerator="cpu", devices=1)
   # On a machine with a supported CUDA GPU:
   # model.fit(X_train, y_train, accelerator="gpu", devices=1)

Native solvers have their own device controls; see the architecture guide.

Batch Size
----------

Batching is controlled via the ``batch_size`` argument in ``fit()``:

.. code-block:: python

   model.fit(X_train, y_train, max_epochs=100, batch_size=128)

Inference can be chunked independently to bound accelerator memory:

.. code-block:: python

   predictions = model.predict(X_test, batch_size=2048)
   components = model.predict_components(X_test, batch_size=2048)

The same option is available on ``predict_proba`` and fixed-basis
``basis_transform`` calls.

Architecture-native training
----------------------------

Most neural architectures use Lightning epochs. IGANN regression and binary
classification instead follow its released sequential ELM algorithm, so
``n_estimators`` is the maximum number of boosting stages and
``early_stopping`` is validation-stage patience. Multiclass and IGANNLSS use
Lightning over the complete fixed ELM basis. The usual NAMpy split,
preprocessing, persistence, prediction batching, and component APIs apply to
both routes. For native fits, inspect attempted stage losses with
``model.training_history()`` and the retained stage count with
``model.n_estimators_``.

Regularization
--------------

Weight Decay (L2)
~~~~~~~~~~~~~~~~~

Add L2 regularization:

.. code-block:: python

   model = NAMRegressor(
       weight_decay=1e-5         # L2 penalty
   )

Dropout
~~~~~~~

Add dropout for regularization:

.. code-block:: python

   model = NAMRegressor(
       dropout=0.3               # Drop 30% of activations
   )

Estimator parameters and sample weights
---------------------------------------

Constructor parameters describe an unfitted estimator. A search can clone it
and set a candidate's parameters without carrying over fitted state:

.. code-block:: python

   from sklearn.base import clone
   from nampy.models import NAMRegressor

   template = NAMRegressor(layer_sizes=[8])
   candidate = clone(template).set_params(dropout=0.05)
   parameters = candidate.get_params(deep=False)

Sample weights change each training row's contribution to the fitting
objective; they are different from a fixed offset in the predictor.
Using the numerical inputs from :doc:`../notebooks/01_getting_started`:

.. code-block:: python

   import numpy as np
   from nampy.models import GAMRegressor

   sample_weight = np.where(X_train["x1"].to_numpy() > 0, 2.0, 1.0)
   weighted_gam = GAMRegressor(k=8, basis="cr").fit(
       X_train[["x1", "x2"]], y_train, sample_weight=sample_weight
   )

Hyperparameter Tuning
----------------------

Manual Search
~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   best_score = -np.inf
   best_params = None

   for lr in [1e-5, 1e-4, 1e-3]:
       for dropout in [0.1, 0.3, 0.5]:
           model = NAMRegressor(dropout=dropout)
           model.fit(X_train, y_train, max_epochs=50, lr=lr)
           score = model.score(X_val, y_val)

           if score > best_score:
               best_score = score
               best_params = {'lr': lr, 'dropout': dropout}

   print(f"Best params: {best_params}")

Use validation rows to choose hyperparameters and checkpoints. Keep test rows
outside both decisions, and fit any external preprocessing inside each
training fold. A good regression score does not establish probability or
interval calibration. Compare probability predictions and reliability in
:doc:`../notebooks/05_real_data_classification`, or quantile coverage and
interval width in :doc:`../in_depth/quantile_calibration`.

Cross-validation
~~~~~~~~~~~~~~~~

Cross-validation evaluates a fitting procedure across several train–validation splits. Any transformation learned from the data must be fitted separately within each training fold.

Use cross-validation to choose settings on the development data, then evaluate the selected procedure on a separate test set. This example uses only the numerical columns ``x1`` and ``x2``; the group feature is excluded.

.. code-block:: python

   from sklearn.model_selection import train_test_split
   from nampy.models import GAMRegressor

   from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

   X_development, X_test, y_development, y_test = train_test_split(
       X[["x1", "x2"]], y, test_size=0.2, random_state=7
   )
   numeric_X = X_development[["x1", "x2"]]
   model = GAMRegressor(optimize_smoothing=False, smoothing_method="fixed", smoothing_params=[0.7, 0.7])
   cv = KFold(n_splits=10, shuffle=True, random_state=7)
   cv_scores = cross_val_score(model, numeric_X, y_development, cv=cv, scoring="r2")

   search = GridSearchCV(
       model,
       {"k": [6, 9]},
       cv=cv,
       scoring="neg_mean_squared_error",
   ).fit(numeric_X, y_development)

   pd.DataFrame({
       "Mean CV R2": [float(cv_scores.mean())],
       "Selected basis dimension": [search.best_params_["k"]],
   }).style.hide(axis="index").format({"Mean CV R2": "{:.3f}"})

The selected model is refit on the development rows. Evaluate
``search.best_estimator_`` on ``X_test`` and ``y_test`` once to report final
performance.

Troubleshooting
---------------

.. list-table::
   :header-rows: 1

   * - Symptom
     - Checks
     - Possible adjustment
   * - Loss does not decrease
     - Check input/target scales, finite values, and loss history.
     - Try a smaller learning rate or more epochs.
   * - Training improves while validation deteriorates
     - Check the split and checkpoint criterion.
     - Use early stopping, dropout, or weight decay.
   * - Both training and validation fit poorly
     - Check preprocessing, optimization, and model capacity.
     - Increase capacity or reduce excessive regularization.
   * - Training is slow
     - Profile batch time and check the selected device.
     - Use an available GPU, larger batches, or a smaller architecture.

Next Steps
----------

* Learn about :doc:`preprocessing`
* Check :doc:`interpretability`
