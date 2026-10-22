{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% set architecture = objname | replace("Regressor", "") | replace("Classifier", "") | replace("LSS", "") %}
{% set guide = architecture | lower %}
{% if architecture == "EnsembleTreeNAM" %}{% set guide = "ensemble_treenam" %}{% endif %}
{% if architecture == "NBMSPAM" %}{% set guide = "nbm_spam" %}{% endif %}
{% if architecture == "SplineNAM" %}{% set guide = "spline_nam" %}{% endif %}
{% if architecture == "GAM" %}
Constructor choices and defaults are explained in the :doc:`statistical GAM
guide <../../user_guide/statistical_models>` and listed in the class signature
above.
{% else %}
Architecture-specific parameter types, allowed values, and defaults are listed
by :class:`nampy.neural.configs.Default{{ architecture }}Config`. The
:doc:`{{ architecture }} guide <../../models/{{ guide }}>`
explains which of those parameters materially change the model.
{% endif %}

{% set essential = ["fit", "predict", "predict_proba", "predict_link", "predict_point", "predict_components", "standard_errors", "score", "evaluate", "explain_terms", "term_importance", "plot", "plot_terms", "plot_interactions", "plot_distributional_effects", "save_model", "load_model"] %}
{% set ns = namespace(found=false) %}
{% for item in methods %}{% if item in essential %}{% set ns.found = true %}{% endif %}{% endfor %}
{% if ns.found %}
Essential methods
-----------------

.. autosummary::
{% for item in methods %}
{% if item in essential %}
   ~{{ objname }}.{{ item }}
{% endif %}
{% endfor %}
{% endif %}

See also
--------

Use the :doc:`estimator overview <../models>` to compare tasks, the
:doc:`user guide <../../user_guide>` for modeling decisions, and the
:doc:`example gallery <../../examples/index>` for complete analyses.

.. minigallery:: nampy.{{ objname }}
   :add-heading: Related examples
