Tutorials
=========

Learn the shared estimator workflow, fit and diagnose a statistical GAM,
compare neural representations, and then model conditional spread. These
walkthroughs introduce the main ideas through code and fitted results.

Start with :doc:`your first additive models <../notebooks/01_getting_started>`,
then :doc:`the real-data classification workflow <../notebooks/05_real_data_classification>`.
The first gives a core regression path with optional extensions; the second
connects preprocessing, probability evaluation, interpretation, and reuse.

After that foundation, choose a branch:

* Statistical: :doc:`first_gam` → :doc:`../notebooks/02_statistical_gams`.
* Neural: :doc:`gam_nam` → :doc:`../notebooks/03_neural_models`.
* Distributional: :doc:`changing_variability` → :doc:`../notebooks/04_distributional_models`.
* Specializations: :doc:`../in_depth/index` and :doc:`../howto/index`.
* Extension work: :doc:`../user_guide/custom_models`.

Detailed examples continue these branches with deeper checks; they are not
prerequisites for the core tutorials.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Your first additive models
      :link: ../notebooks/01_getting_started
      :link-type: doc

      Start with fitting, prediction, additive components, scikit-learn integration, and cross-validation.

      :download:`Download notebook <../notebooks/01_getting_started.ipynb>`

   .. grid-item-card:: Classify a real dataset
      :link: ../notebooks/05_real_data_classification
      :link-type: doc

      Handle missing values, compare a logistic baseline, inspect probabilities,
      and save/reload the complete preprocessing and prediction workflow.

   .. grid-item-card:: Fit, diagnose, and extend a statistical GAM
      :link: first_gam
      :link-type: doc

      Follow an energy-demand analysis, then extend it with shape constraints and interaction surfaces.

   .. grid-item-card:: Compare GAM, NAM, and shared-basis models
      :link: gam_nam
      :link-type: doc

      Compare statistical and neural feature effects, then see how shared neural bases change the representation.

   .. grid-item-card:: Learn conditional location and spread
      :link: changing_variability
      :link-type: doc

      Move from predicting a response to modeling how both its location and variability depend on features.

.. toctree::
   :maxdepth: 1
   :hidden:

   Your first additive models <../notebooks/01_getting_started>
   Classify a real dataset <../notebooks/05_real_data_classification>
   Fit, diagnose, and extend a statistical GAM <first_gam>
   Compare GAM, NAM, and shared-basis models <gam_nam>
   Learn conditional location and spread <changing_variability>

For broader theory and longer repository examples, continue with
:doc:`../in_depth/index`.
