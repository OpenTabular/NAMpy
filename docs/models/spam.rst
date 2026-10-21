SPAM
====

When to use it
--------------

Use SPAM when scalable low-rank polynomial interaction structure is more
important than a strictly feature-wise decomposition.

Model structure
---------------

SPAM represents higher-order effects with low-rank homogeneous polynomials.
For degree :math:`d`, a rank-:math:`R_d` block is built from projected inputs,

.. math::

   p_d(x) = \sum_{r=1}^{R_d} a_{dr}
   \left(w_{dr}^{\top}x\right)^d.

The factorized form avoids enumerating every monomial explicitly. NAMpy can
retain unary terms, add lower-order corrections, constrain factors through the
proximal option, and report local expanded-term importance.

Fit and inspect
---------------

.. code-block:: python

   from nampy.models import SPAMRegressor

   model = SPAMRegressor(
       ranks=[64, 32],
       ignore_unary=False,
       lower_order_correction=True,
   ).fit(X_train, y_train)

   prediction = model.predict(X_test)
   components = model.predict_components(X_test, center=True)
   local = model.local_term_importance(X_test.iloc[:5], top_k=8)

Interpretation and limits
-------------------------

Degree blocks summarize polynomial structure. Expanded local terms are
sample-specific and should not be read as globally stable interactions without
additional checks.

Available estimators
--------------------

* :class:`nampy.models.SPAMRegressor`
* :class:`nampy.models.SPAMClassifier`
* :class:`nampy.models.SPAMLSS`

Reference
---------

Dubey, Radenovic, and Mahajan (2022), `Scalable Interpretability via
Polynomials <https://arxiv.org/abs/2205.14108>`_.

