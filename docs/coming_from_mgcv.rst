Coming from mgcv
================

NAMpy provides a Python-native statistical GAM engine whose methodological and
compatibility reference is ``mgcv``. This page helps translate common tasks;
the separate :doc:`compatibility` page documents precisely what has been
implemented and compared.

Common task equivalents
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 32 34 34

   * - Task
     - ``mgcv``
     - NAMpy
   * - Fit a Gaussian GAM
     - ``gam(y ~ s(x), data=d)``
     - ``GAM('y ~ s(x)').fit(d)``
   * - Select smoothness with REML
     - ``method='REML'``
     - ``optimize_smoothing=True, smoothing_method='reml'``
   * - Estimator-style REML fit
     - Not an ``mgcv`` convention
     - ``GAMRegressor(formula=...).fit(d)``
   * - Inspect fitted terms
     - ``plot(fit)``
     - ``fit.plot()`` or ``adapter.plot()``
   * - Check basis dimensions
     - ``k.check(fit)``
     - ``fit.k_check()``
   * - Inspect concurvity
     - ``concurvity(fit)``
     - ``fit.concurvity()``

Important default difference
----------------------------

:class:`nampy.models.GAMRegressor` is designed for estimator workflows and
selects smoothness with REML by default. The lower-level
:class:`nampy.gam.GAM` starts with fixed smoothing unless automatic selection
is requested. Both accept formulas. This distinction is about the level of
control and defaults; formula use is not exclusive to either interface.

The returned objects also follow Python conventions rather than reproducing R
objects. Use :meth:`nampy.gam.GAM.predict` for response, link, term, or matrix
predictions, and use :meth:`nampy.gam.GAM.predict_components` when you need the
shared additive result representation.

Where behavior differs
----------------------

Numerical optimizer traces, Python estimator conventions, persistence, and a
small number of guarded combinations intentionally differ. Do not infer
support from similar syntax alone. Consult :doc:`compatibility` for upstream
versions, tested quantities and tolerances, and explicit unsupported cases.

Next steps
----------

* :doc:`auto_examples/statistical/plot_first_gam` develops a full analysis.
* :doc:`user_guide/statistical_models` explains practical model choices.
* :ref:`statistical-gam-tools` is the detailed capability inventory.
* :doc:`api/gam` is the exact API reference.
