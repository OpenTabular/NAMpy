Notebook Tutorials
==================

``docs/notebooks/`` is the single canonical notebook collection. The four
self-contained notebooks are checked in with fast-mode outputs and rendered by
nbsphinx without being regenerated during the documentation build.

Start with the common workflow, use the complete statistical GAM guide as a
reference, explore neural architectures in the model zoo, and finish with the
registry-driven task and distributional report:

.. toctree::
   :maxdepth: 1
   :caption: Theory-first tutorials

   notebooks/01_nampy_core_workflow
   notebooks/02_complete_gam_guide
   notebooks/03_neural_additive_model_zoo
   notebooks/04_tasks_distributional_models_and_ensembles

The notebooks default to ``FAST_MODE = True`` so every checked-in code cell can
run with a modest sample and training budget. Disable fast mode for substantive
experiments; the fast-mode comparison tables are interface demonstrations, not
competitive benchmarks.

Run locally
-----------

.. code-block:: bash

   pip install -e ".[all,docs]" jupyterlab
   jupyter lab docs/notebooks

Short copy-and-paste recipes remain in :doc:`examples/index`; terminal-runnable
verification programs remain in the root ``examples/`` directory.
