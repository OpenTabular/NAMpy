Neighborhood inputs
-------------------

Configure ``nei`` on the model. It is a mapping with the following entries:

.. list-table::
   :header-rows: 1

   * - Entry
     - Input and interpretation
   * - ``a``
     - Concatenated observation indices to delete in each neighborhood.
   * - ``ma``
     - Cumulative end positions in ``a``, one per neighborhood. These are counts,
       unaffected by ``index_base``; they must be strictly increasing.
   * - ``d``
     - Concatenated observation indices to predict after each deletion.
   * - ``md``
     - Cumulative end positions in ``d``, matching the number of ``ma`` entries.
   * - ``index_base``
     - Observation index convention: 1 by default, or 0 for Python indices.
   * - ``jackknife``
     - Post-fit neighborhood/jackknife option; defaults to -1.
   * - ``gamma``
     - Post-fit neighborhood evaluation multiplier, default 1.0. The selection
       criterion uses model ``score_gamma``, separately.

``nei=None`` constructs leave-one-out neighborhoods. If ``a`` or ``ma`` is
missing, the normalizer restores the leave-one-out structure. When deletion
neighborhoods correspond one-to-one with training rows, omitted ``d``/``md``
can be inferred. Otherwise supply both prediction-index arrays explicitly.

.. code-block:: python

   import numpy as np
   from nampy.gam import normalize_nei
   from nampy.models import GAMRegressor

   n = 100
   nei = normalize_nei({
       'a': np.arange(n), 'ma': np.arange(1, n + 1),
       'd': np.arange(n), 'md': np.arange(1, n + 1),
       'index_base': 0,
   }, n=n)
   model = GAMRegressor(smoothing_method='ncv', nei=nei)
   # model.fit(X_train, y_train)  # inputs must contain n rows

Normalizer signature and output
-------------------------------

.. autofunction:: nampy.gam.normalize_nei
   :no-index:

``n`` is the number of training observations. The keyword ``index_base``, when
provided, overrides the mapping entry. Returns a dictionary containing normalized
integer arrays ``a``, ``ma``, ``d`` and ``md``, defaults for auxiliary options,
and ``index_base=0``. Invalid dimensions, out-of-range observations and mismatched
neighborhood counts raise ``ValueError``; non-mapping inputs raise ``TypeError``.

The criterion uses approximate deletion updates, retaining known offsets and
signed Newton weights. It is not an exact refit for each neighborhood. Its scalar
score and log-smoothing gradient are implemented by the PIRLS NCV backend;
available outer optimizers depend on the derivative support of that route.
