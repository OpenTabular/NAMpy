Can a feature effect be constrained to a defensible shape?
==========================================================

A dose, an exposure, or a risk score is sometimes known to move in one direction before any curve is estimated. A shape constraint encodes that knowledge in the basis. It is not a smoother drawn through the plot afterwards. The construction follows Pya and Wood (2015), as cited in the :ref:`shape-constrained guide <howto-user-guide-shape-constrained-gams>`.

The comparisons use simulated curves with noise. They are not patient records. Use a constraint only when the direction or the curvature is justified for the problem. The :doc:`breast-cancer tutorial </notebooks/05_real_data_classification>` is a separate classification analysis and does not impose monotonicity.

What is the problem?
--------------------

The question is whether the fitted effect can decrease, in a region where a decrease would not be an acceptable description, just because a few noisy observations pull it down. The guide fits the same simulated series twice and compares the curves and the first derivative.

Why is a linear model insufficient?
-----------------------------------

A straight line is already monotone, but it forces a constant slope. The scientific constraint is weaker than linearity: the effect may rise quickly and then flatten, as long as it does not turn down. An unconstrained smooth can represent that shape and can also invent a local decrease. The comparison that matches the constraint is therefore an unconstrained P-spline of the same dimension, not a straight line.

Why this model?
---------------

Both fits in the shape-constrained guide use P-splines with dimension 12, a Gaussian family, and GCV with the BFGS smoothing optimizer:

.. code-block:: python

   GAM(formula='y ~ s(x, bs="ps", k=12)', family="gaussian",
       optimize_smoothing=True, smoothing_method="gcv",
       smoothing_optimizer="bfgs").fit(data)

   GAM(formula='y ~ s(x, bs="mpi", k=12)', family="gaussian",
       optimize_smoothing=True, smoothing_method="gcv",
       smoothing_optimizer="bfgs").fit(data)

``ps`` is the ordinary P-spline and can turn either way. ``mpi`` imposes a nonnegative first derivative, so the smooth is monotone increasing. ``mpd`` is the decreasing counterpart. A count response can use the same ``mpi`` basis with a count family. The guide's documented route for these transformed-coefficient models is GCV. Automatic ML, REML, or LAML selection for them is unsupported.

After the fit, ``increasing.derivative(smooth_number=1, deriv=1)`` returns the first derivative and its Bayesian standard error. That derivative is an estimand of the constrained model. Checking it over the fitted range confirms the basis that was requested.

The :doc:`curvature tutorial </notebooks/curvature_constraints>` repeats the idea for a convex dose response. It draws 240 values of ``x`` uniformly on 0 to 1, with seed 109, from the mean ``0.3 + 2(x − 0.3)²`` plus Gaussian noise of standard deviation 0.2. The second derivative of that mean is 4. Both fits use dimension 12 and GCV with BFGS. ``ps`` is unconstrained. ``cx`` imposes nonnegative curvature.

.. image:: /_static/applications_convex.png
   :alt: Ordinary and convex smooths on a quadratic dose response, with their second derivatives and a reference line at curvature 4
   :class: scientific-figure
   :align: center

The left panel is the mean response. Points are the simulated doses, the dashed curve is ``0.3 + 2(x − 0.3)²``, and the two solid curves are the ordinary and convex fits. The right panel is the second derivative at the training locations. The dashed horizontal line is zero and the dotted line is the generating curvature, 4. The convex fit stays on the nonnegative side of zero. The notebook records that this fitted curvature need not equal 4: the basis imposes the sign, and the data set the shape inside that restriction. A nonnegative second derivative is the convex constraint, in the same way a nonnegative first derivative is the increasing constraint.

What the fit showed
-------------------

On the noisy monotone example, the unconstrained curve turns downward locally. The monotone fit does not: its first derivative stays nonnegative on the fitted range. The comparison figure is the two curves on the same noisy series. The derivative figure is the constrained first derivative, a pointwise uncertainty band, and a horizontal line at zero. The band sitting above zero is the fitted constraint holding on that range. It is not a test that monotonicity is true in the population.

.. image:: /_static/shape_constraint_comparison.png
   :alt: Unconstrained and monotone-increasing smooths fitted to noisy data
   :class: scientific-figure
   :align: center

.. image:: /_static/shape_constraint_derivative.png
   :alt: Fitted monotone smooth derivative with a pointwise uncertainty band and zero reference
   :class: scientific-figure
   :align: center

Limits of the interpretation
----------------------------

If the true effect is not monotone, the constraint hides that structure and the fit can still look smooth. A constrained dose curve from a simulation does not transfer to a clinical covariate whose monotonicity was not established before fitting. The reason for the constraint belongs in the analysis before ``mpi`` is chosen. Changing ``k`` from 12 to 10, as the guide also illustrates, changes the available flexibility. It does not relax the sign of the derivative.

:download:`Download the convex-dose notebook <../notebooks/curvature_constraints.ipynb>`
