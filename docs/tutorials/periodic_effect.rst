Model a Periodic Effect
=======================

Hour 0 and hour 24 describe the same daily boundary. An ordinary spline can
estimate different values or slopes at those endpoints; a cyclic cubic basis
joins them by construction.

Here ``hour`` is time on a 24-hour clock, ``temperature`` is outdoor
temperature in degrees Celsius, and ``demand`` is electricity demand in
megawatts. The :doc:`first GAM example
<../auto_examples/statistical/plot_first_gam>` generates ``train`` and explains
the full synthetic energy study.

.. code-block:: python

   from nampy.models import GAMRegressor

   model = GAMRegressor(
       formula=(
           'demand ~ s(temperature, bs="cr", k=12) '
           '+ s(hour, bs="cc", k=12)'
       ),
       family="gaussian",
   ).fit(train)

   figures = model.plot(
       pages=1, rug=False, scheme=1,
       shade_col="#f3b4c2", scale=0,
   )

.. image:: ../_static/first_gam_effects.png
   :alt: Estimated temperature and cyclic hour effects
   :class: scientific-figure
   :align: center

The hour curve meets at the daily boundary. ``k`` supplies representational
capacity; REML chooses the effective smoothness within that capacity. The
basis-dimension diagnostic in the :doc:`first GAM example
<../auto_examples/statistical/plot_first_gam>` checks whether the chosen
capacity appears adequate.
