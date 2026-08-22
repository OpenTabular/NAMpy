Preprocessing (PreTab)
======================

NAMpy does not implement its own tabular preprocessing. All preprocessing is done by the **PreTab** library. You pass a PreTab preprocessor instance into the data module or into the sklearn-style models (e.g. :class:`nampy.models.nam.NAMRegressor`).

Using PreTab with NAMpy
-----------------------

1. Create a PreTab preprocessor with the options you need (task, output_dim, numerical/categorical strategies, etc.). See the `PreTab documentation <https://pypi.org/project/pretab/>`_ for full options.

2. Use it either:

   * **With the high-level sklearn-style API** — pass preprocessor-related keyword arguments when constructing the model; NAMpy will build a PreTab preprocessor internally and use it in ``fit``:

   .. code-block:: python

      from nampy.models import NAMRegressor

      model = NAMRegressor(
          task="regression",
          output_dim=50,
          numerical_method="ple",  # or other PreTab options
      )
      model.fit(X_train, y_train, max_epochs=100)
      predictions = model.predict(X_test)

   * **With the data module directly** — build a PreTab preprocessor yourself and pass it to :class:`nampy.neural.data.datamodule.NAMpyDataModule`:

   .. code-block:: python

      from pretab.preprocessor import Preprocessor
      from nampy.neural.data.datamodule import NAMpyDataModule

      preprocessor = Preprocessor(task="regression", output_dim=50)
      data_module = NAMpyDataModule(
          preprocessor=preprocessor,
          batch_size=128,
          shuffle=True,
          regression=True,
      )
      data_module.setup_data(X_train, y_train, val_size=0.2)
      # Use data_module with Lightning Trainer and NAMpy base models

Preprocessor contract
---------------------

Any preprocessor used with NAMpy (including PreTab’s) must:

* Implement ``fit(X, y)`` and ``transform(X)``.
* Expose ``get_feature_info(verbose=False)`` returning a tuple ``(num_feature_info, cat_feature_info, emb_feature_info)`` — three dicts of feature names to metadata (e.g. ``preprocessing``, ``dimension``) so that NAMpy can build the right input tensors.

Accessing the preprocessor
--------------------------

After fitting a sklearn-style model, the preprocessor is available as ``model.preprocessor`` (the PreTab instance). You can use it to transform new data or inspect feature info:

.. code-block:: python

   preprocessor = model.preprocessor
   X_transformed = preprocessor.transform(X_new)
   num_info, cat_info, emb_info = preprocessor.get_feature_info(verbose=False)

For full PreTab options and API, see the `PreTab project <https://pypi.org/project/pretab/>`_.
NAMpy's supported contract and the model-specific consequences of grouped
feature blocks are documented in :doc:`pretab_compatibility`.
