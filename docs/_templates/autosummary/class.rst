{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% if module == "nampy.models" and not objname.startswith("Neural") %}
.. include:: /_generated/reference/{{ objname }}.inc
{% endif %}
