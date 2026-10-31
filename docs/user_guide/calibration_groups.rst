Calibrate fixed strata and complete groups
==========================================

Choose the exchangeable unit before splitting data. Stratum and complete-group
coverage are different targets.

Fixed strata and complete groups
--------------------------------

``calibrate_grouped`` and ``calibrate_regressor(..., groups=...)`` distinguish
two targets; group IDs are never treated as frequency weights:

* ``group_mode="stratum"`` (array-level ``mode="stratum"``): apply separate
  quantiles within an externally fixed covariate partition. Coverage is within
  a stratum under within-stratum exchangeability. Small strata can give
  unbounded sets; unseen strata raise without pooled fallback. Outcome-defined
  or calibration-tuned partitions are unsupported.
* ``group_mode="cluster"`` (default): take the maximum nonconformity score over
  all members of each complete calibration group, then calibrate across those
  group scores. Each group contributes once, irrespective of its size. Coverage
  is simultaneous over all members of one new exchangeable complete group,
  marginal over groups. It is not simultaneous across all future groups and
  does not describe an additional member of a previously seen group.

Pass ``groups=group_ids_test`` to ``predict_interval``. Cluster-mode query IDs
must be disjoint from calibration group IDs. The caller must also keep training,
validation and calibration groups separate and supply complete groups. Group
sizes and their observation mechanisms are part of the exchangeability
assumption. The software cannot certify that a supplied trajectory is complete.
Stratum membership must use the same fixed rule at calibration and prediction.
``calibration_counts`` records the relevant calibration count per query row.
Combining density-ratio weighting with grouping, hierarchical random-member
methods, partial groups and temporal dependence are not implemented.

For methodological citations and paper links, see :doc:`/references`.
