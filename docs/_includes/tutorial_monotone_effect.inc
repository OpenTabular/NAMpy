.. _tutorial-monotone:

Constrain a smooth with prior knowledge
---------------------------------------

Suppose prior knowledge says that a dose effect may level off but cannot
decrease over the studied range. The formula below makes that assumption part
of estimation by replacing an ordinary cubic basis with ``bs="mpi"``.

.. code-block:: python

   import numpy as np
   import pandas as pd

   rng = np.random.default_rng(9)
   dose = np.sort(rng.uniform(0.0, 10.0, 180))
   signal = 1.0 + 4.0 / (1.0 + np.exp(-(dose - 5.0)))
   data = pd.DataFrame({"dose": dose, "response": signal + rng.normal(0.0, 0.45, dose.size)})


   from nampy.gam import GAM

   ordinary = GAM(
       formula='response ~ s(dose, bs="cr", k=12)',
       family="gaussian",
       optimize_smoothing=True,
       smoothing_method="reml",
   ).fit(data)

   increasing = GAM(
       formula='response ~ s(dose, bs="mpi", k=12)',
       family="gaussian",
       optimize_smoothing=True,
       smoothing_method="gcv",
       smoothing_optimizer="bfgs",
   ).fit(data)

   derivative = increasing.derivative(smooth_number=1, deriv=1)

.. code-block:: python

   import matplotlib.pyplot as plt

   grid = pd.DataFrame({"dose": np.linspace(0.0, 10.0, 240)})
   plt.scatter(data["dose"], data["response"], s=13, alpha=0.4, color="gray")
   plt.plot(grid["dose"], ordinary.predict(grid, type="response"), label="Unconstrained")
   plt.plot(grid["dose"], increasing.predict(grid, type="response"), label="Monotone increasing")
   plt.xlabel("Dose")
   plt.ylabel("Expected response")
   plt.legend()

.. image:: ../_static/shape_constraint_comparison.png
   :alt: Ordinary and monotone-increasing fits to noisy observations
   :class: scientific-figure
   :align: center

The ordinary smooth follows local noise and briefly turns downward. The
constrained fit does not. Its first derivative is nonnegative over the fitted
range. This restriction is useful only when the monotonicity assumption is
defensible; it should not be selected merely because the resulting curve
looks simpler. See :doc:`../user_guide/shape_constrained_gams` for the full
basis and fitting reference.
