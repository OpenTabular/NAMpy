Training Models
===============

This guide covers how to train nampy models effectively.

Basic Training
--------------

All nampy models follow the same training interface:

.. code-block:: python

   from nampy.models import NAMRegressor
   
   model = NAMRegressor()
   model.fit(X_train, y_train, max_epochs=100, lr=1e-3)

Training Parameters
-------------------

Common Training Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Complete Example
~~~~~~~~~~~~~~~~

.. code-block:: python

   model.fit(
       X_train,
       y_train,
       max_epochs=150,           # Train for up to 150 epochs
       lr=1e-4,                  # Learning rate
       patience=10,              # Stop if no improvement for 10 epochs
       val_size=0.2              # Use 20% for validation
   )

Early Stopping
--------------

Early stopping prevents overfitting by monitoring validation performance:

.. code-block:: python

   model.fit(
       X_train,
       y_train,
       max_epochs=200,
       patience=15,              # Stop if val loss doesn't improve for 15 epochs
       val_size=0.2
   )

Benefits:
* Prevents overfitting
* Saves training time
* Finds optimal stopping point automatically

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

Step-wise schedules are selected at fit time. ``cosine`` reproduces the
schedule used by the NBM-SPAM reference trainer; adding warm-up steps yields
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

nampy automatically uses GPU if available:

.. code-block:: python

   import torch
   
   # Check GPU availability
   print(f"CUDA available: {torch.cuda.is_available()}")
   
   # Train on GPU (automatic)
   model = NAMRegressor()
   model.fit(X_train, y_train, max_epochs=100)

For CPU-only training, set:

.. code-block:: python

   import os
   os.environ['CUDA_VISIBLE_DEVICES'] = ''

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

Hyperparameter Tuning
----------------------

Manual Search
~~~~~~~~~~~~~

.. code-block:: python

   best_score = 0
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

Using scikit-learn
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.model_selection import cross_val_score
   
   model = NAMRegressor()
   scores = cross_val_score(
       model, X, y, cv=5,
       fit_params={'max_epochs': 50, 'lr': 1e-3}
   )
   
   print(f"CV Score: {scores.mean():.4f} (+/- {scores.std():.4f})")

Training Tips
-------------

1. **Start with defaults** - nampy's defaults work well for most cases
2. **Use early stopping** - Set `patience` to avoid overfitting
3. **Monitor validation loss** - Ensure the model is learning
4. **Try different learning rates** - Start with 1e-4, adjust as needed
5. **Use GPU** - Significantly speeds up training
6. **Validate on held-out data** - Don't overfit to training set

Common Issues
-------------

Model Not Converging
~~~~~~~~~~~~~~~~~~~~

* **Reduce learning rate**: Try 1e-5 instead of 1e-4
* **Increase max_epochs**: Give the model more time
* **Check data**: Ensure features and target are properly scaled

Overfitting
~~~~~~~~~~~

* **Increase dropout**: Try 0.5 instead of 0.3
* **Add weight_decay**: Try 1e-5 or 1e-4
* **Use early stopping**: Set patience=10
* **Get more data**: If possible

Underfitting
~~~~~~~~~~~~

* **Increase model capacity**: Larger `layer_sizes`
* **Train longer**: More `max_epochs`
* **Reduce regularization**: Lower dropout, weight_decay
* **Try different preprocessing**: PLE often works well

Slow Training
~~~~~~~~~~~~~

* **Use GPU**: Check `torch.cuda.is_available()`
* **Reduce max_epochs**: Start with 50 for experimentation
* **Use smaller model**: Reduce `layer_sizes`
* **Reduce output_dim**: Fewer bins = faster training

Next Steps
----------

* Learn about :doc:`preprocessing`
* Implement :doc:`custom_models`
* Check :doc:`interpretability`
