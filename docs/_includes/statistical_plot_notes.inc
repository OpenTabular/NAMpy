Derivative curves and region maps
---------------------------------

``model.plot_derivative(smooth_number=1, deriv=1, level=0.95)`` displays a
supported univariate smooth's slope and pointwise Bayesian band. Use
``deriv=2`` for curvature. It returns matplotlib figures and is also available
on the estimator adapters. By default it sorts the observed covariate values;
``X=`` supplies evaluation rows where the derivative API permits them. Bands
are conditional, not simultaneous, and derivatives are on the additive
predictor scale. Formula-transformed covariates retain their transformed axis.
The existing basis restrictions apply: ``ps`` and ``bs`` provide derivative
matrices; ``cr``, GP and SOS do not. SCAM derivatives remain training-data only.

``model.plot()`` renders an MRF region map when its smooth has
``xt={"polys": polygons}``. ``polygons`` maps region names to two-column
coordinate arrays; NaN rows separate disconnected loops or nested holes.
Map values follow the polygon names, including when their order differs from
factor-level order. Scheme 0 uses heat colors and other schemes use gray.
MRF smooths supplied only with a neighbor graph or penalty have no polygon
geometry and are omitted from plots. Matplotlib rendering is not pixel-identical
to R graphics; the plotted fitted effects are checked against ``mgcv``.

Conditional prediction surfaces
-------------------------------

``vis`` displays a full-model prediction surface while holding other predictors
fixed. It differs from a term plot, which displays an individual smooth's
contribution.

.. code-block:: python

   surface = model.vis(
       view=["x", "z"], cond={"group": "control"},
       type="response", plot_type="contour", n_grid=40, too_far=0.1,
   )
   figures = surface["figures"]
   predictions = surface["fit"]

The default view uses the first two varying predictors. Other numeric predictors
use mgcv's observed type-3 median; factors use their most frequent level.
``cond`` overrides those values. Factor views are supported; distance masking
with ``too_far`` requires numeric view variables. ``lp`` selects a one-based
linear predictor for multi-predictor fits.

Use ``plot_type="persp"`` for a perspective surface. A positive ``se`` adds
plus/minus that many pointwise standard errors; requesting contour plots with
positive ``se`` warns and draws perspective surfaces, following mgcv. ``color``
accepts ``heat``, ``topo``, ``cm``, ``terrain``, ``gray`` or ``bw``. The returned
``fit`` and ``se_fit`` arrays vary the first view fastest; reshape them to
``(n_grid, n_grid)`` for rows along the second view and columns along the first.
Use ``prepare_vis_gam_data`` from ``nampy.gam.diagnostics`` to obtain data without
rendering. Estimator adapters also expose ``vis``.

Spherical hemisphere plots
--------------------------

``model.plot(scheme=0, theta=30, phi=30)`` draws a rotated SOS hemisphere with
colored fitted effects, latitude/longitude lines, contours and visible
observations. Scheme 1 gives a contour-only hemisphere. Coordinates remain
latitude then longitude in degrees; ``theta`` and ``phi`` set the viewing
orientation. ``n2`` controls the disk resolution. Schemes 2 and above delegate
to the ordinary rectangular smooth plots with the scheme reduced by two, as
in mgcv. Hemisphere plots do not add standard-error bands, and SOS derivative
matrices remain unavailable.

These displays use Matplotlib; numerical projection and prediction data are
compared against mgcv, while graphics are not claimed to be pixel-identical.

QQ and four-panel model checks
------------------------------

``model.qq(rep=200, level=0.9, seed=7)`` plots observed residuals against
family-based reference quantiles, with a pointwise simulation envelope.
``rep=0`` averages randomized quantile calculations where the family supplies a
quantile function; otherwise it uses 50 simulations without an envelope.
``level>=1`` draws individual simulated curves. These are diagnostic envelopes,
not simultaneous confidence bands or a calibrated goodness-of-fit test.

``model.plot_check(type="deviance", seed=7)`` draws the QQ panel, residuals
against the linear predictor, a residual histogram using R's bin rules, and
response against fitted values. ``old_style=True`` uses normal QQ quantiles.
Both methods return prepared numerical data plus a ``figures`` list and are
also available on the sklearn-style GAM adapters. ``model.gam_check()`` retains
its numerical convergence/rank/basis-check report.

Use ``prepare_qq_gam_data`` and ``prepare_gam_check_data`` from
``nampy.gam.diagnostics`` for plotting-free data. Quantile and simulation support
covers the implemented ordinary, extended and location-scale families; families
without either capability use the upstream normal-QQ fallback. A NumPy ``seed``
is reproducible but does not reproduce R's RNG stream. Exact reference tests
supply ``quantile_permutations`` (zero-based, shape ``(s_rep, n)``) or
``response_draws`` (shape ``(rep, n)``) to both implementations.

Shape-constrained fits automatically use SCAM display semantics: one-dimensional
term plots default to two standard errors, two-dimensional plots to one;
partial residuals use SCAM's normalized square-root IRLS weights. Bivariate
shape plots center on their plotting grid; ``vis`` continues to show full-model
predictions. Shape uncertainty uses transformed coefficient covariance;
``se_with_mean`` does not alter shape-term uncertainty. ``plot_check`` additionally
accepts ``type="rquantile"`` for SCAM fits. The explicit preparation functions
``prepare_plot_scam_data``, ``prepare_vis_scam_data``, ``prepare_qq_scam_data`` and
``prepare_scam_check_data`` are available for custom displays. SCAM check panels
do not run a k-index test. Graphics use Matplotlib and do not promise identical
R pixels.
