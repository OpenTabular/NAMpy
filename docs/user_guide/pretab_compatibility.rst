:orphan:

Published PreTab compatibility
==============================

NAMpy targets the public block-level preprocessing contract in the published
PreTab ``0.0.3`` release. It does not require the experimental preprocessing
extensions that were previously carried in NAMpy's reference checkout of
PreTab.

This boundary is intentional. PreTab owns generic tabular preprocessing, while
NAMpy owns the interpretation of the resulting feature blocks inside each
neural architecture. The neural architecture implementations are not replaced
or approximated as part of this change.

Supported contract
------------------

NAMpy relies on the following published PreTab 0.0.3 behavior:

* ``Preprocessor.fit(X, y)`` and ``Preprocessor.transform(X)``;
* dictionary output with ``num_<feature>`` and ``cat_<feature>`` blocks;
* ``get_feature_info(verbose=False)`` metadata, particularly ``dimension``,
  ``categories``, and ``preprocessing``;
* PreTab's numerical-then-categorical block ordering;
* standard constructor options such as ``numerical_preprocessing``,
  ``categorical_preprocessing``, ``feature_preprocessing``, ``n_bins``,
  ``n_knots``, ``degree``, ``scaling_strategy``, and ``task``.

High-level NAMpy estimators discover the supported constructor parameters from
the installed ``Preprocessor``. Unsupported names continue to raise an explicit
``TypeError`` rather than being ignored.

Model impact
------------

Fitted piecewise-linear encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NAMpy freezes the learned piecewise-linear thresholds after PreTab fitting.
PreTab 0.0.3 otherwise adjusts the last threshold to the maximum of each query
batch, which makes a row's encoding depend on unrelated rows and batch size.
The fitted adapter preserves the learned basis, cut-point and extrapolation
semantics while making prediction, checkpoint reloads and Fisher derivatives
independent of query batching. An encoder with no learned thresholds is rejected
explicitly; an unsplittable numerical tree can still fail during upstream fitting.

NAM
~~~

NAM creates one feature network per PreTab block. A numerical source
feature normally produces a scalar block. A one-hot categorical source feature
produces one multi-column block and therefore one network for the categorical
feature as a whole. This differs from the earlier parity configuration, which
created a separate network for every one-hot column, but it is a valid grouped
categorical NAM and remains additively interpretable at source-feature level.

NBM
~~~

NBM already flattens block dimensions into scalar concepts internally. For
example, a three-column ``cat_group`` block becomes the concepts
``group[0]``, ``group[1]``, and ``group[2]``. Dense, sparse, Conv1D, einsum,
and n-ary NBM implementations are unchanged. The removed surfaces affect input
scaling, names, and ordering relative to the reference implementation, not the
NBM basis-network core.

SPAM and NBM-SPAM
~~~~~~~~~~~~~~~~~

SPAM and NBM-SPAM use the same internal block flattening as NBM. Their
polynomial cores, ranks, penalties, proximal behavior, and hybrid block
assembly are unchanged. Terms now follow PreTab's block order.

NodeGAM
~~~~~~~

NodeGAM can continue to select ``numerical_preprocessing="quantile"``. It no longer
requests NODE-GAM-specific quantile noise or quantile-transform tuning through
PreTab, so preprocessing will not reproduce the reference repository exactly.
The differentiable tree architecture and its training controls are unchanged.

Other neural models
~~~~~~~~~~~~~~~~~~~

LinReg, SNAM, SIAN, GPNAM, IGANN, NATT, NAMformer, TreeNAM,
EnsembleTreeNAM, QNAM, and SplineNAM already consume PreTab's
block-level contract. Their architecture implementations required no changes
for this compatibility decision.

GAM backend
~~~~~~~~~~~

``nampy.gam`` does not import or use PreTab and is completely unaffected.
