:orphan:

PreTab migration history
========================

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
   consumes PreTab's numerical-then-categorical block order.

``output_scaling`` and ``output_range``
   NAMpy no longer requests a second scaler over the fully encoded matrix.
   NAM, NBM, SPAM, and NBM-SPAM use PreTab's supported numerical
   ``scaling_strategy="minmax"`` preprocessing instead. This scaler maps numerical
   inputs to ``[-1, 1]``; categorical one-hot values remain ``0`` or ``1``.

Quantile fit controls
   ``quantile_n_quantiles``, ``quantile_output_distribution``, and
   ``quantile_noise`` are no longer exposed by NAMpy examples or required by
   its tests. ``numerical_preprocessing="quantile"`` remains supported with
   PreTab's own behavior.

Generic representation parameters
   NAMpy no longer relies on ``representation_params`` or mapping-valued
   per-feature specifications such as
   ``{"x": {"method": "...", ...}}``. PreTab's supported
   ``feature_preprocessing`` form remains available.

TF-IDF categorical representation
   The extended ``tfidf`` representation is no longer advertised or tested.
   Text features must use a representation supported by the published PreTab
   release or be prepared before they reach NAMpy.

Atomic transformed-column metadata
   NAMpy no longer expects ``output_index`` metadata or synthesized metadata
   entries for individual encoded columns. Block dictionary order and each
   block's ``dimension`` are the canonical layout contract.

The current supported interface is documented in :doc:`../user_guide/pretab_compatibility`.
