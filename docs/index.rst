NAMpy: Interpretable Additive Modeling
======================================

.. image:: _static/logo.png
   :alt: NAMpy — Interpretable Additive Modeling
   :align: center
   :width: 704px

NAMpy combines a strict ``mgcv``-aligned statistical GAM backend with neural
additive models for regression, classification, and distributional regression.

Choose a starting point
-----------------------

* :doc:`quickstart` — install NAMpy and fit one statistical GAM and one neural
  additive model.
* :doc:`notebook_tutorials` — theory-first, executable tutorials for every
  supported model family.
* :doc:`user_guide` — preprocessing, training, interpretation, shape
  constraints, and extension contracts.
* :doc:`models/index` — a compact model-selection catalog.
* :doc:`examples/index` — short task recipes and standalone verification
  scripts.
* :doc:`api/index` — constructor, method, and function signatures.

Documentation roles
-------------------

Each kind of documentation has one owner. Tutorials and model theory live in
``docs/notebooks/`` and are regenerated before every build. Sphinx user guides explain concepts shared
by multiple models. The root ``examples/`` directory contains longer scripts
that can be run from a terminal. API pages are generated from docstrings and
do not repeat tutorials.

.. toctree::
   :maxdepth: 2
   :caption: Learn and use NAMpy

   installation
   quickstart
   notebook_tutorials
   models/index
   user_guide
   examples/index
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Project information

   architecture
   contributing
   license
   faq
   development/reference_sources
   development/mgcv_optimizer_boundaries

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
