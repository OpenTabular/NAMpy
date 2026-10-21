Tutorials
=========

These executable notebooks introduce NAMpy through complete analyses. Start
with the common workflow, then choose the model family or task relevant to your
work.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Getting started with NAMpy
      :link: notebooks/01_getting_started
      :link-type: doc

      Fit models, generate predictions, inspect additive components, and save
      fitted estimators.

   .. grid-item-card:: Statistical GAMs
      :link: notebooks/02_statistical_gams
      :link-type: doc

      Work with formulas, smooth bases, smoothing selection, inference,
      diagnostics, and shape constraints.

   .. grid-item-card:: Neural models
      :link: notebooks/03_neural_models
      :link-type: doc

      Compare neural feature functions and interaction structures through a
      common estimator interface.

   .. grid-item-card:: Distributional models and ensembles
      :link: notebooks/04_distributional_models
      :link-type: doc

      Fit classification, distributional, quantile, and ensemble models and
      interpret their outputs.

.. toctree::
   :maxdepth: 1
   :hidden:

   notebooks/01_getting_started
   notebooks/02_statistical_gams
   notebooks/03_neural_models
   notebooks/04_distributional_models

The rendered pages include the saved outputs. To modify and rerun a notebook,
open its ``.ipynb`` source from the repository. For shorter examples focused on
one task, see :doc:`examples/index`.
