Extended Notebooks
==================

These notebooks provide broad, executable surveys of NAMpy. For a shorter
analysis organized around one modeling question, start with the
:doc:`tutorial gallery <tutorials/index>`.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Getting started with NAMpy
      :link: notebooks/01_getting_started
      :link-type: doc

      Fit models, generate predictions, inspect additive components, and save
      fitted estimators.

   .. grid-item-card:: Statistical GAM reference notebook
      :link: notebooks/02_statistical_gams
      :link-type: doc

      Survey formulas, smooth bases, smoothing selection, inference,
      diagnostics, and shape constraints in one place.

   .. grid-item-card:: Neural models
      :link: notebooks/03_neural_models
      :link-type: doc

      Compare neural feature functions and interaction structures through a
      common estimator interface.

   .. grid-item-card:: Distributional capability notebook
      :link: notebooks/04_distributional_models
      :link-type: doc

      Survey classification, distributional, quantile, and ensemble surfaces.

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
