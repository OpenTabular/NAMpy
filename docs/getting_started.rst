Getting Started
===============

Install NAMpy, complete one first analysis, then choose a model from the result
you need. The two introductory tutorials use the same synthetic energy-demand
data so their fitted effects can be compared directly.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      Install NAMpy and verify that the package is available.

   .. grid-item-card:: Fit your first GAM
      :link: quickstart
      :link-type: doc

      Fit smooth effects, evaluate held-out predictions, and inspect
      uncertainty and diagnostics.

   .. grid-item-card:: Fit your first neural additive model
      :link: neural_quickstart
      :link-type: doc

      Learn neural feature functions and interpret their centered
      contributions.

   .. grid-item-card:: Choose a model
      :link: models/index
      :link-type: doc

      Start from inference, neural feature learning, or distributional
      modeling before comparing specialized architectures.

.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   quickstart
   neural_quickstart
   models/index
