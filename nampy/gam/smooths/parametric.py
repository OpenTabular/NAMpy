from .smooth_base import BaseSmoothTerm, _resolve_feature, column_as_float


class LinearTerm(BaseSmoothTerm):
    term_type = "parametric"
    basis_name = "linear"
    supports_tensor_marginal = False

    def __init__(self, feature, label=None, term_id=None, metadata=None):
        super().__init__(
            feature=feature,
            label=label,
            term_id=term_id,
            smoothing_id=None,
            by=None,
            metadata=metadata,
        )
        self._feature_index = None
        self._feature_name = None
        self._basis_train = None

    def fit(self, X, feature_names):
        idx, feature_name = _resolve_feature(self.feature, feature_names)
        self._feature_index = idx
        self._feature_name = feature_name
        self._set_resolved_features([feature_name])
        self._basis_train = column_as_float(X, idx)[:, None]
        self._penalties = []
        self._record_constraint_result(None, None, absorbed_by=None)
        return self

    @property
    def n_coef(self):
        self._require_fitted()
        return 1

    def transform_new(self, X_new):
        self._require_fitted()
        return column_as_float(X_new, self._feature_index)[:, None]


__all__ = ["LinearTerm"]
