{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {%- if all_methods %}
   .. rubric:: Methods

   .. autosummary::
   {%- for item in all_methods %}
   {%- if item != '__init__' %}
      ~{{ objname }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {%- endif %}
   
   

   
   {%- if all_attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {%- for item in all_attributes %}
      ~{{ objname }}.{{ item }}
   {%- endfor %}
   {%- endif %}
   
