Preprocessing (PreTab)
======================

Neural estimators use **PreTab** for tabular preprocessing. Configure their
preprocessing through constructor keywords, or pass a PreTab instance to the
low-level neural data module. The statistical GAM backend builds its own
formula/design matrices and does not use PreTab.

Configure preprocessing through the estimator
----------------------------------------------

Start with estimator keywords. The estimator fits transformations using its
training rows and reuses them for validation and prediction; do not fit a new
preprocessor on prediction rows. The small example below contains numerical
age and a two-level category. Its final row is new input.

.. include:: /_generated/howto_preprocessing_input.inc

.. literalinclude:: ../scripts/generate_howto_assets.py
   :language: python
   :start-after: # PREPROCESSING START
   :end-before: # PREPROCESSING PLOTTING
   :dedent: 4

.. dropdown:: Plotting code

   .. literalinclude:: ../scripts/generate_howto_assets.py
      :language: python
      :start-after: # PREPROCESSING PLOTTING
      :end-before: # PREPROCESSING END
      :dedent: 4

.. include:: /_generated/howto_preprocessing_blocks.inc

.. image:: /_static/howto_preprocessing.png
   :alt: Feature blocks grouped by original feature, with a separate new-row transformation
   :class: scientific-figure

Age occupies one transformed column; the category occupies a one-hot block.
The dashed boundary separates the new row. The six supplied training rows
define the fitted age transformation and category vocabulary. The explicit
validation rows and the new row reuse that fitted state; neither updates it.

Use PreTab directly with a data module
--------------------------------------

For a custom Lightning workflow, supply a PreTab instance to the data module:

.. code-block:: python

   from pretab.preprocessor import Preprocessor
   from nampy.neural.data.datamodule import NAMpyDataModule

   preprocessor = Preprocessor(task="regression", n_bins=50)
   data_module = NAMpyDataModule(
       preprocessor=preprocessor, batch_size=128,
       shuffle=True, regression=True,
   )
   data_module.setup_data(X_train, y_train, val_size=0.2)

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

Custom preprocessing interfaces are described in :doc:`custom_models`.
