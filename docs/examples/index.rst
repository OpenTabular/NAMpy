Examples
========

This is the single index for NAMpy examples. Each entry is labeled by format
so you can choose the depth you need without first navigating separate
``tutorials``, ``recipes``, and ``notebooks`` sections.

Generated gallery
-----------------

The flagship examples execute during the documentation build. Their figures
and printed outputs are captured automatically, and each page provides
downloadable Python and notebook versions.

.. toctree::
   :maxdepth: 2

   ../auto_examples/index

Statistical modeling
--------------------

* **Tutorial:** :doc:`First statistical GAM
  <../auto_examples/statistical/plot_first_gam>` — fit, evaluate, interpret,
  and diagnose one model.
* **Tutorial:** :doc:`Monotone effect
  <../auto_examples/statistical/plot_monotone_effect>` — compare an
  unconstrained smooth with a scientifically justified increasing effect.
* **Tutorial:** :doc:`Cyclic effect <../tutorials/periodic_effect>` — encode a
  periodic boundary.
* **Tutorial:** :doc:`Interaction surface <../tutorials/interaction_surface>`
  — interpret an effect that depends jointly on two inputs.
* **Recipe:** :doc:`Use an offset <../howto/offsets>`.

Neural modeling
---------------

* **Tutorial:** :doc:`First neural additive model
  <../auto_examples/neural/plot_first_nam>` — train a NAM and inspect its
  feature effects.
* **Tutorial:** :doc:`NAM versus NBM <../tutorials/nam_nbm>` — compare two
  representations on the same data and scale.
* **Recipe:** :doc:`Regression <basic_regression>`.
* **Recipe:** :doc:`Classification <basic_classification>`.

Distributional modeling
-----------------------

* **Tutorial:** :doc:`Changing conditional variability
  <../auto_examples/distributional/plot_changing_variability>` — model
  location and scale separately.
* **Recipe:** :doc:`Distributional regression <distributional_regression>`.

Extended notebooks
------------------

The notebooks are broad executable surveys rather than first introductions.
Use them after the focused examples above.

* **Extended notebook:** :doc:`Getting started <../notebooks/01_getting_started>`.
* **Extended notebook:** :doc:`Statistical GAMs <../notebooks/02_statistical_gams>`.
* **Extended notebook:** :doc:`Neural models <../notebooks/03_neural_models>`.
* **Extended notebook:** :doc:`Distributional models
  <../notebooks/04_distributional_models>`.

.. toctree::
   :maxdepth: 1
   :hidden:

   basic_regression
   basic_classification
   distributional_regression
   ../tutorials/periodic_effect
   ../tutorials/interaction_surface
   ../tutorials/nam_nbm
   ../howto/interactions
   ../howto/monotonicity
   ../howto/offsets
   ../howto/persistence
   ../notebooks/01_getting_started
   ../notebooks/02_statistical_gams
   ../notebooks/03_neural_models
   ../notebooks/04_distributional_models
