Pristine PreTab compatibility
=============================

NAMpy targets the public block-level preprocessing contract in pristine PreTab
``1.0.0rc2``. It does not require the experimental preprocessing extensions
that were previously carried in NAMpy's reference checkout of PreTab.

This boundary is intentional. PreTab owns generic tabular preprocessing, while
NAMpy owns the interpretation of the resulting feature blocks inside each
neural architecture. The neural architecture implementations are not replaced
or approximated as part of this change.

Supported contract
------------------

NAMpy relies on the following pristine PreTab behavior:

* ``Preprocessor.fit(X, y)`` and ``Preprocessor.transform(X)``;
* dictionary output with ``num_<feature>`` and ``cat_<feature>`` blocks;
* ``get_feature_info(verbose=False)`` metadata, particularly ``dimension``,
  ``categories``, and ``preprocessing``;
* pristine PreTab's numerical-then-categorical block ordering;
* standard constructor options such as ``numerical_method``,
  ``categorical_method``, ``feature_preprocessing``, ``output_dim``, ``degree``,
  ``scaling``, ``random_state``, and ``dtype``.

High-level NAMpy estimators discover the supported constructor parameters from
the installed ``Preprocessor``. Unsupported names continue to raise an explicit
``TypeError`` rather than being ignored.

Removed requirements
--------------------

The following extended PreTab surfaces are no longer required or configured by
NAMpy:

``output_granularity``
   NAMpy no longer asks PreTab to split every encoded output column into a
   separate dictionary entry. Multi-column representations remain grouped by
   source feature.

``output_order``
   NAMpy no longer requests categorical-first or original-input ordering. It
   consumes pristine PreTab's numerical-then-categorical block order.

``output_scaling`` and ``output_range``
   NAMpy no longer requests a second scaler over the fully encoded matrix.
   NAM, NBM, SPAM, and NBM-SPAM use pristine PreTab's supported numerical
   ``scaling="minmax"`` preprocessing instead. This scaler maps numerical
   inputs to ``[-1, 1]``; categorical one-hot values remain ``0`` or ``1``.

Quantile fit controls
   ``quantile_n_quantiles``, ``quantile_output_distribution``, and
   ``quantile_noise`` are no longer exposed by NAMpy examples or required by
   its tests. ``numerical_method="quantile"`` remains supported with pristine
   PreTab's own behavior.

Generic representation parameters
   NAMpy no longer relies on ``representation_params`` or mapping-valued
   per-feature specifications such as
   ``{"x": {"method": "...", ...}}``. Pristine PreTab's supported
   ``feature_preprocessing`` form remains available.

TF-IDF categorical representation
   The extended ``tfidf`` representation is no longer advertised or tested.
   Text features must use a representation supported by the installed pristine
   PreTab release or be prepared before they reach NAMpy.

Atomic transformed-column metadata
   NAMpy no longer expects ``output_index`` metadata or synthesized metadata
   entries for individual encoded columns. Block dictionary order and each
   block's ``dimension`` are the canonical layout contract.

Model impact
------------

NAM
~~~

NAM creates one feature network per pristine PreTab block. A numerical source
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
assembly are unchanged. Terms now follow pristine PreTab's block order.

NodeGAM
~~~~~~~

NodeGAM can continue to select ``numerical_method="quantile"``. It no longer
requests NODE-GAM-specific quantile noise or quantile-transform tuning through
PreTab, so preprocessing will not reproduce the reference repository exactly.
The differentiable tree architecture and its training controls are unchanged.

Other neural models
~~~~~~~~~~~~~~~~~~~

LinReg, SNAM, SIAN, GPNAM, IGANN, NATT, NAMformer, TreeNAM,
EnsembleTreeNAM, QNAM, and SplineNAM already consume pristine PreTab's
block-level contract. Their architecture implementations required no changes
for this compatibility decision.

GAM backend
~~~~~~~~~~~

``nampy.gam`` does not import or use PreTab and is completely unaffected.

Behavioral consequences
------------------------

This compatibility target guarantees that the supported neural estimators can
be constructed, cloned, fitted, and used with pristine PreTab. It does not
claim preprocessing parity with every model's reference repository. In
particular, encoded scaling, categorical NAM topology, transformed-column
ordering, and optional quantile or TF-IDF representations can differ.

Users needing an unsupported representation should request it in PreTab rather
than adding a second generic preprocessing implementation to NAMpy.
