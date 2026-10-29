Location and scale contract
---------------------------

``link`` is a length-two tuple/list. The mean link accepts ``identity``,
``log``, ``inverse`` or ``sqrt``. The second link must be ``logb``. With
``b=0.01`` and second predictor ``eta_sigma``,

.. math::

   \sigma = b + \exp(\eta_\sigma), \qquad \tau=1/\sigma.

Thus ``b`` is a lower bound on standard deviation, and the internal family
parameter is precision ``tau``. The natural-parameter interface exposes standard
deviation, not precision. A log-scale predictor must not be interpreted as an
additive change in response variance.

``distribution_parameters_from_eta(eta)`` takes an ``(n_samples, 2)`` NumPy
array and returns ``(n_samples, 2)`` columns ``(mu, sigma)``.
``distribution_parameter_jacobian(eta)`` returns the two per-row diagonal
transformation derivatives. With the default mean link they are ``1`` and
``exp(eta[:, 1])``. Full uncertainty propagation also needs coefficient
cross-covariance between the two predictors.

``GAMLSS.predict_parameters`` returns natural ``(mu, sigma)`` columns;
``GAMLSS.predict_link`` returns the two link predictors.
``GAMLSS.predict`` returns ``mu``. Source-style ``GAM`` response predictions
retain the family's ``(mu, tau)`` convention, so use the chosen interface's
contract rather than assuming the second column is always standard deviation.

Example
-------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from nampy.models import GAMLSS

   rng = np.random.default_rng(7)
   X = pd.DataFrame({'x': np.linspace(-1, 1, 120)})
   y = np.sin(3 * X.x) + rng.normal(size=len(X)) * (0.2 + 0.2 * (X.x + 1))
   model = GAMLSS(
       family='gaulss',
       formula={'mu': 'y ~ s(x, k=8)', 'sigma': '~ s(x, k=6)'},
   ).fit(X, y)
   parameters = model.predict_parameters(X)   # (120, 2): mu, sigma
   raw = model.predict_link(X)
   mean = model.predict(X)   # (120,)

The family uses a joint general-family solve rather than ordinary single-mean
PIRLS. ML/REML support does not imply all combinations with coefficient
constraints, custom penalties or smoothing optimizers are available.
