Generalized Additive Models
===========================

When to use them
----------------

Use a statistical GAM when smoothness selection, model-based uncertainty,
effective degrees of freedom, term tests, and residual diagnostics are part of
the analysis.

Model structure
---------------

A generalized linear model makes a transformed response mean linear in its
inputs. A GAM replaces selected straight-line terms with functions estimated
from data, while retaining an additive predictor:

.. math::

   Y_i\mid x_i\sim\mathcal F(\mu_i,\phi),\qquad
   g(\mu_i)=\eta_i=\beta_0+z_i^\mathsf T\alpha
   +\sum_j f_j(x_{ij})+o_i.

Here the **family** specifies response support and variability, the **link** g
connects the conditional mean to the predictor, and the **offset** o is a known
contribution. The parametric part z-transpose alpha can contain linear effects
and factor contrasts. A smooth may depend on one input or on several jointly;
additivity is between terms, not a prohibition on explicit interactions.

For example, a Poisson log-link model with
``count ~ s(time) + temperature + offset(log(exposure))`` describes expected
count as exposure times an exponentiated additive rate predictor. A change
delta in a smooth multiplies the conditional rate by exp(delta). With a
Gaussian identity link, the same delta is an additive conditional-mean change.
Thus a term plot must be interpreted on its documented link scale.

How a smooth is learned
~~~~~~~~~~~~~~~~~~~~~~~

NAMpy represents each function with a finite collection of basis functions:

.. math::

   f_j(x)=\sum_{r=1}^{k_j}b_{jr}(x)\theta_{jr},\qquad
   \widehat\theta=\arg\min_\theta\left\{
   -\ell(\theta)+\frac12\sum_h\lambda_h\theta^\mathsf TS_h\theta\right\}.

The basis determines the functions available to fit; its penalty matrix S
measures their roughness. For a cubic spline, roughness is integrated squared
curvature. Lambda balances the evidence in the response against that
roughness. This regularization prevents a generous basis from simply
interpolating noise. **Basis dimension k and smoothing strength lambda have
different roles:** k limits capacity, while lambda controls its use.

An ordinary curvature penalty leaves straight lines unpenalized. Heavy
smoothing can therefore leave a trend rather than delete the effect.
Shrinkage bases or additional null-space penalties permit whole-term
shrinkage. Ordinary centering constraints separate a smooth's constant
direction from the intercept; a centered value of zero is a reference
convention rather than an absence of association.

See :doc:`../statistical_reference/smooths/theory` for the design-matrix
derivation, null spaces, centering, tensor products and effective degrees of
freedom. Individual basis pages explain how their assumptions differ.

Selecting smoothness and measuring uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At fixed lambda, the backend estimates coefficients: Gaussian models can use
penalized least squares, ordinary non-Gaussian models use penalized IRLS, and
general distributional models use joint likelihood derivatives. Automatic
smoothness selection adds an outer problem that repeatedly evaluates those
fits. GCV/UBRE approximate predictive risk; ML/REML integrate penalized
directions using a random-effects interpretation, with Laplace approximation
where required. Newton or BFGS is an optimization algorithm for a criterion,
not another criterion. The :doc:`smoothness theory
<../statistical_reference/selection/theory>` explains this distinction.

Effective degrees of freedom measure fitted flexibility rather than simply
counting basis columns. Coefficient covariance propagates to a linear
predictor through its design row:

.. math::

   \operatorname{Var}\{\widehat\eta(x)\}\approx
   b(x)^\mathsf TV_\theta b(x).

Pointwise mean uncertainty differs from the variability of a future response,
and pointwise intervals are not simultaneous bands over an entire curve.
Standard covariance usually conditions on estimated smoothing parameters;
corrections require support from the fitted route. See
:doc:`../statistical_reference/inference` for the available conventions.

Extensions of the same model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor terms model joint surfaces, factor smooths model group-dependent curves,
and SCOP bases encode monotonicity or curvature restrictions. Distributional
GAMs give several distribution parameters their own predictors and fit them
jointly. These choices change the mathematical model; they are not merely
different renderings of the same univariate mean curve. The
:doc:`family theory <../statistical_reference/families/theory>` explains the
connection between response laws, links and parameter-specific predictors.

Fit and inspect
---------------

.. code-block:: python

   from nampy.gam import GAM

   model = GAM(
       formula="y ~ s(x, bs='tp', k=15) + s(z, bs='cr', k=10)",
       family="gaussian",
       optimize_smoothing=True,
       smoothing_method="reml",
   ).fit(data=df)

   prediction, se = model.predict(df, return_se=True)
   terms = model.predict_terms(df)
   model.summary()
   model.plot(residuals=True)

For distributional regression, provide one formula per parameter:

.. code-block:: python

   from nampy.models import GAMLSS

   model = GAMLSS(
       family="normal",
       formula={"mu": "y ~ s(x)", "sigma": "~ s(z)"},
   ).fit(df)
   parameters = model.predict_parameters(df)


Interpretation and limits
-------------------------

Use a formula when the basis, tensor-product structure, offsets, or shape
constraints should be explicit. Both ``GAM`` and the estimator adapters accept
formulas. ``GAMRegressor`` and ``GAMClassifier`` add estimator conventions and
automatic REML smoothing by default; raw ``GAM`` starts with fixed smoothing
unless selection is requested.

Available interfaces
--------------------

* :class:`nampy.gam.GAM`
* :class:`nampy.models.GAMRegressor`
* :class:`nampy.models.GAMClassifier`
* :class:`nampy.models.GAMLSS`

See :doc:`../user_guide/statistical_gams` for a runnable analysis workflow and
:doc:`../architecture` for the seven stages that implement it.

References
----------

* Hastie and Tibshirani (1986), `Generalized Additive Models
  <https://doi.org/10.1214/ss/1177013604>`_.
* Wood (2017), *Generalized Additive Models: An Introduction with R*, second
  edition.


* Wood, Pya, and Säfken (2016), `Smoothing Parameter and Model Selection
  for General Smooth Models <https://doi.org/10.1080/01621459.2016.1180986>`_.
