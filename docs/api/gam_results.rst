GAM Results
===========

Structured outputs from the statistical fitting and reporting layers. These
objects keep numerical solver state separate from the concise, user-facing
views returned by :class:`nampy.gam.GAM`.

Which result object to use
--------------------------

* ``FitResult`` holds the numerical outputs shared by solver backends.
* ``FitCoreSolution`` pairs that record with transient fitting workspace.
* ``GAMFitResult``, returned by ``model.fit_result()``, adds per-term summaries,
  smoothing parameters, and criterion information around the numerical record.
* ``GAMResult`` aggregates the compiled model, core solution, and reporting
  summary. It is the fitted-state container used across the model lifecycle.

For analysis, start with ``model.fit_result()`` and its ``term_results``;
low-level fitting extensions may need the core solution and aggregate.

.. currentmodule:: nampy.gam.results

.. autosummary::
   :toctree: generated/
   :nosignatures:

   GAMResult
   GAMFitResult
   TermFitResult
   FitResult

