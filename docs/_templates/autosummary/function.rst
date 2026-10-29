{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autofunction:: {{ objname }}

{% if module in ["nampy.inspection", "nampy.plotting", "nampy.gam.formula", "nampy.gam.results"] %}
.. include:: /_generated/reference/symbols/{{ module }}.{{ objname }}.inc
{% endif %}
