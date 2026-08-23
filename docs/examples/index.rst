Examples
========

NAMpy has three deliberately different example formats:

* The pages below are short, copyable task recipes.
* :doc:`../notebook_tutorials` contains theory-first, executable model
  tutorials generated and rendered as part of the documentation build.
* The root `examples directory <https://github.com/Ananyapam7/NAMpy/tree/main/examples>`_
  contains longer standalone verification scripts, including synthetic
  data-generating processes, fitted-model checks, and plots.

.. toctree::
   :maxdepth: 1

   basic_regression
   basic_classification
   distributional_regression

Choosing the right format
-------------------------

Use a recipe when you already know which task you need and want the smallest
working estimator call. Use a notebook when you want mathematical background,
model-specific controls, interpretation, and references. Use a standalone
script when you want a terminal-runnable verification with explicit numerical
checks.

Custom architectures are documented once, in
:doc:`../user_guide/custom_models`. Exact constructors and methods live in the
:doc:`../api/index`.

Running repository material
---------------------------

From the repository root:

.. code-block:: bash

   pip install -e ".[all,docs]" jupyterlab
   jupyter lab docs/notebooks

Standalone scripts can be run directly, for example:

.. code-block:: bash

   python examples/example_gam2.py
   python examples/example_nam.py
