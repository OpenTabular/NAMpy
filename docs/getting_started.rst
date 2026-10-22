Getting started
===============

Start with the modeling result you need. Both introductory examples use the
same synthetic electricity-demand setting, define every variable and unit,
evaluate held-out predictions, and end by interpreting fitted effects.

Install NAMpy
-------------

.. code-block:: bash

   pip install nampy

This command installs the statistical and neural modeling functionality. The
:doc:`installation` page covers supported Python versions and environment
checks. Development installation belongs in :doc:`contributing`.

Choose a first analysis
-----------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Fit your first statistical GAM
      :link: auto_examples/statistical/plot_first_gam
      :link-type: doc

      Use smooth terms, automatic smoothness selection, uncertainty, and
      diagnostics to explain electricity demand.

   .. grid-item-card:: Fit your first neural additive model
      :link: auto_examples/neural/plot_first_nam
      :link-type: doc

      Learn neural feature functions for the same response and inspect them on
      the same scale.

Which model should I start with?
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 58 42

   * - What you want to do
     - Start here
   * - Estimate smooth nonlinear effects with statistical inference and
       diagnostics
     - :doc:`Statistical GAMs <user_guide/statistical_models>`
   * - Learn inspectable nonlinear effects with neural networks
     - :doc:`NAM <auto_examples/neural/plot_first_nam>`
   * - Model changing variability as well as the conditional mean
     - :doc:`Distributional modeling <user_guide/gamlss>`

Once those distinctions are clear, the :doc:`neural model comparison
<models/index>` explains specialized architectures for sparsity, interactions,
shared bases, tree functions, and other structural choices.

Documentation versions
----------------------

The version shown above is the package version imported when the documentation
was built. On Read the Docs, choose ``stable`` for the latest release and
``latest`` for current development documentation. Install the release
documented by ``stable`` with ``pip install nampy``.

.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   faq
