{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% if module == "nampy.models" and not objname.startswith("Neural") %}
.. include:: /_generated/reference/{{ objname }}.inc
{% endif %}

{% if module in ["nampy.contracts", "nampy.plotting", "nampy.gam.formula", "nampy.gam.results"] %}
.. include:: /_generated/reference/symbols/{{ module }}.{{ objname }}.inc
{% endif %}
