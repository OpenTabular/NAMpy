{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autofunction:: {{ objname }}

{% if module in ["nampy.diagnostics", "nampy.plotting", "nampy.gam.formula", "nampy.gam.results"] %}
.. include:: /_generated/reference/symbols/{{ module }}.{{ objname }}.inc
{% endif %}
