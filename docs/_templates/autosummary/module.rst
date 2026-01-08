{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :no-members:
   :no-inherited-members:

   
   .. rubric:: Classes

   .. autosummary::
   {%- for item in all_classes %}
      {{ item }}
   {%- endfor %}
   
