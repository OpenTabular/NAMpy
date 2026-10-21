NAMformer
=========

NAMformer combines transformer-based tabular prediction with feature-level
readout heads. Each input is embedded as a token. A transformer produces a
global prediction from the classification token, while separate linear heads
read the original feature tokens and optional networks read named interaction
embeddings.

.. math::

   \eta(x) = h_{\mathrm{global}}(x)
   + \sum_j m_j(x_j)
   + \sum_{S \in \mathcal{I}} h_S(e_S) + \beta_0.

This dependence structure matters for interpretation: the marginal heads
:math:`m_j` are feature-specific, explicit interaction heads use only their
named embeddings, and the global transformer term depends on all inputs. The
model is therefore not limited to a purely feature-additive predictor.

.. code-block:: python

   from nampy.models import NAMformerRegressor

   model = NAMformerRegressor(
       d_model=64,
       n_layers=4,
       n_heads=4,
       interaction_degree=2,
   ).fit(X_train, y_train)

   prediction = model.predict(X_test)
   marginal_components = model.predict_components(X_test, center=True)

Use the component output to study the explicit marginal and interaction heads;
do not treat it as a unique decomposition of the global transformer term.

Available estimators
--------------------

* :class:`nampy.models.NAMformerRegressor`
* :class:`nampy.models.NAMformerClassifier`
* :class:`nampy.models.NAMformerLSS`

Reference
---------

Thielmann, Reuter, and Säfken (2025), `Beyond Black-Box Predictions:
Identifying Marginal Feature Effects in Tabular Transformer Networks
<https://arxiv.org/abs/2504.08712>`_.

