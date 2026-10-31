Capability inventory
====================

Use this inventory to find an implemented route, then consult its task guide
and the configured model's ``capabilities()`` result. The tables are generated
from the owning family registries and public capability discovery; the
observation and fitting descriptions explain which experiment each route
represents. They do not authorize arbitrary combinations of families, smooths,
optimizers or observation schemes.

Implementation support and validation evidence are separate. A declared
operation can require a fitted model, a supported covariance, explicit offsets
or row context. Evidence below names selected tests and their scope; a reference
fixture or coverage simulation does not validate every supported configuration.

Choose a fitting route
----------------------

.. include:: _generated/capabilities/models.inc

This is a compact selection of public workflows, rather than a second estimator
catalog. Neural architecture availability remains owned by the
:doc:`neural model registry <neural_reference/index>`. For choices within each
task, use :doc:`user_guide/gamlss`, :doc:`user_guide/survival`,
:doc:`user_guide/quantile_inference` and :doc:`user_guide/semistructured`.

Find prediction and uncertainty operations
------------------------------------------

The following statuses describe configured, unfitted models. ``conditional``
means that the listed requirements must be satisfied; fitting is one such
requirement. It does not mean that an uncertainty calculation has already been
performed. Ask the fitted model again to include its actual fitted-state
restrictions.

.. code-block:: python

   from nampy import CalibratedQuantileGAM

   model = CalibratedQuantileGAM(bandwidth=None, optimize_smoothing=True)
   capabilities = model.capabilities()
   capability = capabilities["uncertainty.quantile_function"]
   print(capability.state, capability.conditions)

``uncertainty.quantile_function`` describes calibrated confidence intervals for
an estimated quantile. It is distinct from a future-response interval obtained
by :doc:`conformal calibration <user_guide/calibration>`. Fisher diagnostics
require a declared observation experiment and remain separate from coefficient
covariance or survival-curve intervals.

.. include:: _generated/capabilities/operations.inc

Find a response family
-----------------------

Both registries supply the family rows below. An empty smoothing declaration
means the family does not advertise a particular automatic optimizer here;
it is not a claim that a neural objective selects statistical smoothing
parameters. A probability operation describes the prediction law, not fitting
or an uncertainty method. Family- and architecture-dependent inference must be
queried on the configured/fitted estimator.

Exact, censored and truncated observations are different experiments. For an
observation family, prediction context and the new observation target are
separate inputs: censoring bounds from fitting must not be reused to score
new observations. See :doc:`user_guide/gamlss` for the observation workflow.

.. include:: _generated/capabilities/families.inc

Inspect scoped validation evidence
-----------------------------------

These annotations identify existing test modules. They record neither the
latest CI result nor a guarantee about parameter regimes absent from those
tests. Unannotated families retain their runtime declarations but have no
separate evidence indexed in this compact inventory. Detailed numerical and
upstream comparisons remain in :doc:`development/index`.

.. include:: _generated/capabilities/evidence.inc

The :download:`machine-readable inventory <_generated/capabilities/inventory.json>`
keeps each model/family declaration and its evidence annotations in separate
fields. It is regenerated during the documentation build; update the runtime
owner for implementation support and the evidence annotations only when a
specific test actually establishes that evidence.
