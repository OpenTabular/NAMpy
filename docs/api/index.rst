API Reference
=============

This section is the signature-level reference generated from NAMpy's public
docstrings. For workflows and explanation, use the :doc:`../quickstart`,
:doc:`../user_guide`, or :doc:`../notebook_tutorials` instead.

.. toctree::
   :maxdepth: 2

   models
   gam
   basemodels
   configs
   utils

Public layers
-------------

* :mod:`nampy.models` provides high-level scikit-learn-style estimators,
  including the GAM adapters.
* :mod:`nampy.gam` provides the formula-oriented GAM API and fit core.
* :mod:`nampy.neural` provides architecture, objective, configuration, and
  distribution internals for advanced use and extension.
* :mod:`nampy.contracts` provides backend-neutral prediction, capability,
  feature-schema, and persistence contracts.

Version
-------

.. autodata:: nampy.__version__
   :annotation:
