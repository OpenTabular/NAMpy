Formula API
===========

NAMpy translates a model formula into a typed description of its parametric
terms, smooths, interactions, offsets, and linear predictors. Most analyses
only need to pass the formula to :class:`nampy.gam.GAM`:

.. code-block:: python

   from nampy.gam import GAM

   model = GAM(
       formula='y ~ group + s(x, bs="cr", k=12) + ti(x, z)',
       family="gaussian",
       optimize_smoothing=True,
       smoothing_method="reml",
   ).fit(data)

See :ref:`formula-visual-reference` to choose terms from fitted examples,
:ref:`gam-formula-language` for a worked formula workflow, and
:ref:`gam-smooth-bases` for every public ``bs=`` code.
The objects below are useful when validating formulas or extending the
statistical backend.

Parsing and extraction
----------------------

.. currentmodule:: nampy.gam.formula

.. autosummary::
   :toctree: generated/
   :nosignatures:

   parse_gam_formula
   extract_formula_terms
   all_vars1
   get_numeric_response_labels
   strip_offset_wrapper

Parsed formula objects
----------------------

These records describe the formula before data are attached or basis matrices
are constructed.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ParsedGAMFormula
   ParsedPredictorFormula
   ParsedFormulaComponent
   ParsedParametricTerm
   ParsedSmoothTerm

Extracted term objects
----------------------

Extraction converts parsed syntax into one declarative term collection per
linear predictor.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ExtractedPredictor
   ExtractedParametricTerm
   ExtractedSmoothTerm

