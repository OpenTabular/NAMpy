User Guide
==========

The user guide owns concepts and contracts that apply across model families.
It deliberately does not repeat per-model theory or complete tutorials; those
live in :doc:`notebook_tutorials`.

.. toctree::
   :maxdepth: 2

   user_guide/preprocessing
   user_guide/pretab_compatibility
   user_guide/training
   user_guide/shape_constrained_gams
   user_guide/custom_models
   user_guide/interpretability

Guide map
---------

Preprocessing
~~~~~~~~~~~~~

:doc:`user_guide/preprocessing` explains numerical and categorical transforms.
:doc:`user_guide/pretab_compatibility` records the exact PreTab compatibility
contract and architecture-specific exceptions.

Training
~~~~~~~~

:doc:`user_guide/training` covers validation splits, callbacks, checkpoints,
warm starts, weighting, and optimization controls shared by neural estimators.
Statistical GAM smoothing criteria and optimizers are demonstrated in the GAM
notebook linked from :doc:`notebook_tutorials`.

Interpretation
~~~~~~~~~~~~~~

:doc:`user_guide/interpretability` documents additive components, importance,
and plotting conventions. Every model notebook then demonstrates the relevant
interpretation surface for that architecture.

Shape-constrained GAMs
~~~~~~~~~~~~~~~~~~~~~~

:doc:`user_guide/shape_constrained_gams` is the authoritative list of supported
shape bases, constraints, and current boundaries. The GAM notebook provides a
narrative example rather than duplicating that support matrix.

Extending NAMpy
~~~~~~~~~~~~~~~

:doc:`user_guide/custom_models` is the single complete guide to registering a
custom neural architecture. The API reference supplies signatures for the
classes used by that guide.

Related material
----------------

* :doc:`models/index` — model selection at a glance
* :doc:`examples/index` — short task recipes and runnable scripts
* :doc:`api/index` — exact public API
