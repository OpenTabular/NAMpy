Frequently Asked Questions
==========================

General Questions
-----------------

What is NAMpy?
~~~~~~~~~~~~~~

NAMpy is a Python package for neural additive models and related architectures,
providing interpretable deep learning for tabular data with a scikit-learn compatible interface.

Why use NAMpy over traditional methods?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NAMpy combines the performance of deep learning with the interpretability of
additive models. It offers:

* Better performance than linear models
* More interpretability than black-box neural networks
* Scikit-learn compatible API
* Multiple model architectures for different use cases

Is NAMpy production-ready?
~~~~~~~~~~~~~~~~~~~~~~~~~~

NAMpy 0.1 is a beta release. Validate the estimator, data preprocessing, and
deployment path against your own requirements before production use.

Installation & Setup
--------------------

How do I install NAMpy?
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install "nampy[all]"

For development installation:

.. code-block:: bash

   git clone https://github.com/OpenTabular/NAMpy.git
   cd NAMpy
   pip install -e ".[all,dev]"

What are the requirements?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Python 3.11 or 3.12
The core package requires NumPy, pandas, and scikit-learn. Install
``nampy[gam]`` for the GAM backend, ``nampy[neural]`` for the neural backend,
or ``nampy[all]`` for both.

Do I need a GPU?
~~~~~~~~~~~~~~~~

No, but GPU acceleration is supported and will speed up training if available.

Usage Questions
---------------

How do I choose which model to use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Maximum interpretability**: NAM, NodeGAM
* **Uncertainty quantification**: GPNAM
* **Feature selection**: SNAM
* **Complex interactions**: NATT, NAMformer
* **Baseline**: LinReg

See :doc:`models/index` for detailed comparisons.

Can I use NAMpy with pandas DataFrames?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. NAMpy works with pandas DataFrames:

.. code-block:: python

   import pandas as pd
   from nampy.models import NAMRegressor
   
   df = pd.read_csv("data.csv")
   X = df.drop("target", axis=1)
   y = df["target"]
   
   model = NAMRegressor()
   model.fit(X, y, max_epochs=100)

How do I handle categorical features?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NAMpy's preprocessor handles categorical features automatically. You can also
specify how to treat features:

.. code-block:: python

   model = NAMRegressor(
       cat_cutoff=0.03,  # Treat as categorical if <3% unique values
       treat_all_integers_as_numerical=False
   )

Can I use NAMpy with scikit-learn pipelines?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. NAMpy models are scikit-learn compatible:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from nampy.models import NAMRegressor
   
   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('model', NAMRegressor())
   ])

Performance Questions
---------------------

Why is training slow?
~~~~~~~~~~~~~~~~~~~~~

Deep learning models require more computation. Try:

* Use GPU if available
* Reduce `max_epochs`
* Smaller `layer_sizes`
* Fewer `n_bins`
* Smaller dataset for prototyping

How do I speed up predictions?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Use GPU
* Batch predictions
* Save and load trained models
* Use simpler model architecture

My model is overfitting, what should I do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Increase `dropout`
* Add `weight_decay`
* Use early stopping (`patience` parameter)
* Reduce model complexity
* Get more training data

My model is underfitting, what should I do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Increase model capacity (`layer_sizes`)
* Train longer (`max_epochs`)
* Reduce `dropout`
* Try different model architecture
* Feature engineering

Technical Questions
-------------------

How do I save and load models?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nampy.models import NAMRegressor

   path = model.save_model("nam_model.nampy")
   restored = NAMRegressor.load_model(path)

The file contains the estimator, its fitted neural network, and preprocessing
state. It uses Python's pickle protocol, so only load files from trusted sources.
Recreate persisted files when moving between incompatible Python, PyTorch, or
NAMpy versions.

Can I implement custom models?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! See :doc:`user_guide/custom_models` for a complete guide.

How do I extract feature importances?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For interpretable models like NAM, you can extract feature-level predictions
from the model's forward pass. The exact method depends on the model architecture.

Does NAMpy support multi-output regression?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The regression estimators accept a two-dimensional target and return one column
per output. This is tested for every public neural regressor. Classification and
LSS targets retain their task-specific shapes.

Integration Questions
---------------------

Can I use NAMpy with other frameworks?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NAMpy is built on PyTorch, so it integrates well with the PyTorch ecosystem.
It also works with scikit-learn utilities.

Does NAMpy work with Jupyter notebooks?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! NAMpy works great in Jupyter notebooks. Check out the examples in the
repository's `examples/` directory.

Can I deploy NAMpy models in production?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! NAMpy models can be serialized and deployed like any scikit-learn model.
Consider using model serving frameworks like FastAPI, Flask, or TorchServe.

Contributing Questions
----------------------

How can I contribute?
~~~~~~~~~~~~~~~~~~~~~

See :doc:`contributing` for detailed guidelines.

How do I report a bug?
~~~~~~~~~~~~~~~~~~~~~~

Open an issue on `GitHub <https://github.com/OpenTabular/NAMpy/issues>`_
with:

* Clear description of the problem
* Steps to reproduce
* Expected vs. actual behavior
* Environment details

How do I request a feature?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open an issue on `GitHub <https://github.com/OpenTabular/NAMpy/issues>`_
describing:

* The feature and its motivation
* Use cases
* Possible implementation approach

Still Have Questions?
---------------------

* Check the :doc:`user_guide`
* Browse the :doc:`examples/index`
* Search `GitHub Issues <https://github.com/OpenTabular/NAMpy/issues>`_
* Ask in `GitHub Discussions <https://github.com/OpenTabular/NAMpy/discussions>`_
* Read the :doc:`api/index`
