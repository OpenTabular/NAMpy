NATT
====

When to use it
--------------

Use NATT when additive readouts should be combined with contextual
representations of categorical tokens.

Model structure
---------------

NATT combines additive output terms with tabular-transformer representations.
Numerical features use feature-specific subnetworks. Categorical tokens are
encoded jointly, then token-specific heads contribute to the final sum;
explicit interaction networks can be added for named feature subsets.

.. math::

   \eta(x) = \beta_0 + \sum_j h_j(x) +
   \sum_{S \in \mathcal{I}} h_S(x_S).

For numerical main effects, :math:`h_j` depends on the corresponding feature
block. A categorical token is contextualized by attention, so its named
contribution may depend on the other categorical inputs. Treat those values as
contextual contributions, not as a univariate function :math:`f_j(x_j)`.

Fit and inspect
---------------

.. code-block:: python


   from nampy.neural.configs.natt_config import DefaultNATTConfig
   from nampy.models import NATTRegressor

   model = NATTRegressor(architecture_config=DefaultNATTConfig(d_model=64, n_layers=4, n_heads=4, interaction_degree=2)).fit(X_train, y_train)

   prediction = model.predict(X_test)
   components = model.predict_components(X_test, center=True)
   figures = model.plot_terms(X_test, center=True)

Interpretation and limits
-------------------------

Numerical main effects retain feature-specific inputs. A categorical token is
contextualized by other categorical inputs, so its named contribution is not
generally a univariate function of that token alone.

Available estimators
--------------------

* :class:`nampy.models.NATTRegressor`
* :class:`nampy.models.NATTClassifier`
* :class:`nampy.models.NATTLSS`

Reference
---------

Thielmann et al. (2024), `Interpretable Additive Tabular Transformer Networks
<https://openreview.net/forum?id=TdJ7lpzAkD>`_. Transactions on Machine Learning Research.

