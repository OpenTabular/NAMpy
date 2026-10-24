Frequently Asked Questions
==========================

Installation and setup
----------------------

Which Python versions are supported?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NAMpy supports **Python 3.11 and 3.12**.

How can I verify the installation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Import NAMpy and check its version:

.. code-block:: python

   import nampy

   print(nampy.__version__)

Do I need a GPU?
~~~~~~~~~~~~~~~~

No. Statistical GAMs run on the CPU, and neural estimators also support CPU
training. A GPU is useful when the neural model or dataset is large enough to
offset device-transfer overhead.

How can I check GPU support?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check whether PyTorch can see a CUDA-capable GPU:

.. code-block:: python

   import torch

   print(torch.cuda.is_available())
   if torch.cuda.is_available():
       print(torch.cuda.get_device_name(0))

If CUDA is unavailable, select the appropriate PyTorch build using the
`PyTorch installation guide <https://pytorch.org/get-started/locally/>`_.

Models and data
---------------

Which model should I use?
~~~~~~~~~~~~~~~~~~~~~~~~~

Use the :doc:`model chooser <models/index>` to select a starting architecture
based on your task, interaction structure, and inference needs.

Can I use pandas DataFrames?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. DataFrames preserve column names and dtypes, which makes them the clearest
input format for mixed tabular data.

.. code-block:: python

   import pandas as pd

   from nampy.models import NAMRegressor

   data = pd.read_csv("data.csv")
   X = data.drop(columns="target")
   y = data["target"]

   model = NAMRegressor()
   model.fit(X, y, max_epochs=100)

NumPy arrays are also accepted by the high-level neural estimators.

How are categorical columns detected?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The preprocessing layer uses DataFrame dtypes and estimator settings to assign
feature types. Convert identifier-like integer columns to ``category`` when
they should not be treated as continuous measurements.

.. code-block:: python

   X["region"] = X["region"].astype("category")

See :doc:`user_guide/preprocessing` for the available transformations and
feature-type controls.

What is the difference between a GAM and a neural additive model?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A statistical GAM represents effects with penalized basis functions and selects
smoothness through a statistical criterion. Its output includes quantities such
as standard errors and smooth diagnostics. A neural additive model learns its
feature functions with neural optimization and exposes learned contributions,
but its uncertainty statements are not the same as GAM inference.

Fitting
-------

How can I reduce neural training time?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measure validation performance while changing one setting at a time:

* reduce ``max_epochs`` while prototyping;
* use early stopping;
* reduce the architecture width or number of basis functions;
* increase ``batch_size`` when memory allows; and
* confirm that the intended accelerator is available.

If validation quality matters, do not treat a short demonstration run as a
final fit.

How do I control overfitting?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use a fixed validation split and monitor its metric. For neural models, common
controls include early stopping, ``dropout``, ``weight_decay``, and model
capacity. For statistical GAMs, smoothness selection controls the effective
complexity of smooth terms.

Predictions and interpretation
------------------------------

How do I inspect feature effects?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the fitted estimator's component and plotting methods:

.. code-block:: python

   components = model.predict_components(X_test)
   importance = model.term_importance(X_test)
   figures = model.plot_terms(X_test)

``predict_components`` returns the additive terms used in reconstruction.
``term_importance`` summarizes their magnitude, while ``plot_terms`` displays
the fitted one- and two-dimensional effects supported by the estimator.

For centering and interpretation, see :doc:`user_guide/interpretability`.

What does an LSS model predict?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An ``*LSS`` estimator predicts the natural parameters of its selected
distribution. Their number, order, constraints, and links depend on the family.
Use ``predict_point`` when you need the family-specific point prediction and
consult :doc:`user_guide/gamlss` for parameter semantics.

Does NAMpy support interactions?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, but interaction support depends on the architecture. Some models accept
explicit interaction terms, while others discover or construct higher-order
components. The :doc:`models/index` describes the representation used by each
model.

Persistence and integration
---------------------------

How do I save and load a model?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nampy.models import NAMRegressor

   model.save_model("housing.nampy")
   restored = NAMRegressor.load_model("housing.nampy")

The artifact contains fitted model and preprocessing state. It uses Python's
pickle protocol, so load only trusted files and retain the environment details
used to create long-lived artifacts.

Can I use scikit-learn utilities?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The high-level estimators follow the scikit-learn estimator interface and can
be used with common splitting, scoring, and model-selection utilities. Keep
neural searches sequential when several fits would otherwise compete for the
same GPU memory.

Can I add an architecture?
~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. See :doc:`user_guide/custom_models` for the architecture, configuration,
registry, and estimator contracts.

Troubleshooting and support
---------------------------

Why can Python not import NAMpy after installation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check that the package was installed into the interpreter used by the script or
notebook:

.. code-block:: bash

   python -m pip show nampy

In an editor or notebook, also confirm the selected Python environment or
kernel.

How do I report a problem?
~~~~~~~~~~~~~~~~~~~~~~~~~~

Search the `issue tracker <https://github.com/OpenTabular/NAMpy/issues>`_ first.
For a new report, include a minimal example, the complete error message, the
expected behavior, installation command, and the Python, NAMpy, operating-system,
and backend versions.

How do I contribute?
~~~~~~~~~~~~~~~~~~~~

The :doc:`contributing` page contains the development installation, test
commands, and links to the repository workflow.

Why do new rows fail schema or target checks?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reuse training-time imputers and transforms, and supply the original column
names in the fitted order (``feature_names_in_``). Do not silently sort or drop
columns to make an error disappear. New category handling depends on the fitted
PreTab encoder; test its documented policy before deployment and keep category
construction fixed. See :doc:`user_guide/preprocessing`.

Missing values in the real-data tutorial are imputed using saved training
medians. Targets must also satisfy their family's support: positive continuous
families require positive responses, count families require valid counts, and
probability/composition families impose their own ranges. Consult
:doc:`api/utils` before choosing a family; changing target values merely to
suppress an exception changes the statistical problem.
