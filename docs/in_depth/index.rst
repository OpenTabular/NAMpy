Detailed Examples
=================

Build on the :doc:`introductory tutorials <../tutorials/index>` with broader
modeling theory and longer analyses. The notebooks explore the statistical,
neural, and distributional APIs; the focused examples work through conditional
quantiles and mixed numerical/categorical tree effects using repository scripts.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Statistical GAMs in depth
      :link: ../notebooks/02_statistical_gams
      :link-type: doc

      Select basis capacity with validation data, model group shifts and
      exposure offsets, and fit conditional location and spread.

      :download:`Download notebook <../notebooks/02_statistical_gams.ipynb>`

   .. grid-item-card:: Neural additive architectures
      :link: ../notebooks/03_neural_models
      :link-type: doc

      Independent feature networks, shared learned bases, representation size,
      and a perturbation check of marginal versus global components.

      :download:`Download notebook <../notebooks/03_neural_models.ipynb>`

   .. grid-item-card:: Distributional models and ensemble uncertainty
      :link: ../notebooks/04_distributional_models
      :link-type: doc

      Compare prediction targets, parameter domains, probability metrics,
      conditional quantiles, and disagreement between independent fits.

      :download:`Download notebook <../notebooks/04_distributional_models.ipynb>`

   .. grid-item-card:: Conditional quantiles and interval calibration
      :link: quantile_calibration
      :link-type: doc

      Fit ordered quantiles, compare them with known synthetic truth, check
      coverage, and interpret grouped categorical contributions.

   .. grid-item-card:: Recover step effects with additive trees
      :link: tree_effects
      :link-type: doc

      Compare a single additive tree with a jointly trained tree ensemble on
      numerical thresholds and categorical level shifts.

   .. grid-item-card:: Statistical, neural, and distributional workflows
      :link: ../tutorials/modeling_workflows
      :link-type: doc

      Complete fit-and-plot examples with predictions, components, importance,
      and model-specific figures.

.. toctree::
   :maxdepth: 1
   :hidden:

   ../notebooks/02_statistical_gams
   ../notebooks/03_neural_models
   ../notebooks/04_distributional_models
   quantile_calibration
   tree_effects
   ../tutorials/modeling_workflows
