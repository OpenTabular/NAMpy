Preprocessing (PreTab)
======================

NAMpy does not ship its own preprocessor. All tabular preprocessing is delegated to the **PreTab** library.

Use PreTab's preprocessor when building a :class:`nampy.data_utils.datamodule.NAMpyDataModule` or when using the sklearn-style models (e.g. :class:`nampy.models.nam.NAMRegressor`, :class:`nampy.models.nam.NAMClassifier`). The preprocessor must:

* Implement ``fit(X, y)`` and ``transform(X)``.
* Expose ``get_feature_info(verbose=False)`` returning a tuple ``(num_feature_info, cat_feature_info, emb_feature_info)``, where each element is a dict of feature names to metadata (e.g. ``preprocessing``, ``dimension``).

**Example:**

.. code-block:: python

   from pretab.preprocessor import Preprocessor
   from nampy.data_utils.datamodule import NAMpyDataModule

   preprocessor = Preprocessor(
       task="regression",
       n_bins=50,
       # ... other PreTab options
   )

   data_module = NAMpyDataModule(
       preprocessor=preprocessor,
       batch_size=128,
       shuffle=True,
       regression=True,
   )
   data_module.setup_data(X_train, y_train, val_size=0.2)
   # Then use with Lightning Trainer and NAMpy base models

For full PreTab options and API, see the `PreTab documentation <https://pypi.org/project/pretab/>`_.
