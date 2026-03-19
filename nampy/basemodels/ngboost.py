from ..arch_utils.ngboost_utils import (
    Bernoulli,
    LogNormal,
    LogScore,
    NGBClassifierCore,
    NGBRegressorCore,
    NGBSurvivalCore,
    Normal,
    SUPPORTED_DISTRIBUTIONS,
    build_base_learner,
)
from .boosting_wrapper import (
    BoostingBackendAdapter,
    BoostingSurvivalBackendAdapter,
    GenericBoostingSurvivalWrapper,
    GenericBoostingWrapper,
)
from ..configs.ngboost_config import DefaultNGBoostConfig


def _resolve_distribution(distribution, default, num_classes=None):
    if distribution is None:
        if default == "classification" and num_classes is not None and num_classes > 2:
            from ..arch_utils.ngboost_utils import k_categorical

            return k_categorical(num_classes)
        if default == "classification":
            return Bernoulli
        if default == "survival":
            return LogNormal
        return Normal

    if isinstance(distribution, str):
        key = distribution.lower()
        if key == "categorical":
            from ..arch_utils.ngboost_utils import k_categorical

            if num_classes is None:
                raise ValueError("num_classes is required for categorical NGBoost.")
            return k_categorical(num_classes)
        if key not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"Unsupported distribution {distribution!r}. "
                f"Supported values are: {sorted(SUPPORTED_DISTRIBUTIONS.keys()) + ['categorical']}."
            )
        return SUPPORTED_DISTRIBUTIONS[key]

    return distribution


def _resolve_score(score):
    if score is None:
        return LogScore
    if isinstance(score, str):
        key = score.lower()
        if key != "logscore":
            raise ValueError(f"Unsupported score {score!r}. Only 'logscore' is currently supported.")
        return LogScore
    return score


class NGBoostBackend(BoostingBackendAdapter):
    """NGBoost backend adapter implementing the generic boosting contract."""

    def __init__(
        self,
        task="regression",
        num_classes=None,
        config=DefaultNGBoostConfig(),
        **kwargs,
    ):
        self.task = task
        self.num_classes = num_classes
        self.config = config
        self.hparams = kwargs
        self.estimator = self._build_estimator()

    def _build_estimator(self):
        distribution = _resolve_distribution(
            self.hparams.get("distribution", self.config.distribution),
            default=self.task,
            num_classes=self.num_classes,
        )
        score = _resolve_score(self.hparams.get("score", self.config.score))
        base_estimator = self.hparams.get("Base")
        if base_estimator is None:
            base_estimator = build_base_learner(
                kind=self.hparams.get("base_learner", self.config.base_learner),
                base_kwargs=self.hparams.get(
                    "base_learner_kwargs", self.config.base_learner_kwargs
                ),
            )

        common_kwargs = {
            "Dist": distribution,
            "Score": score,
            "Base": base_estimator,
            "natural_gradient": self.hparams.get(
                "natural_gradient", self.config.natural_gradient
            ),
            "n_estimators": self.hparams.get("n_estimators", self.config.n_estimators),
            "learning_rate": self.hparams.get(
                "learning_rate", self.config.learning_rate
            ),
            "minibatch_frac": self.hparams.get(
                "minibatch_frac", self.config.minibatch_frac
            ),
            "col_sample": self.hparams.get("col_sample", self.config.col_sample),
            "verbose": self.hparams.get("verbose", self.config.verbose),
            "verbose_eval": self.hparams.get("verbose_eval", self.config.verbose_eval),
            "tol": self.hparams.get("tol", self.config.tol),
            "random_state": self.hparams.get("random_state", self.config.random_state),
            "validation_fraction": self.hparams.get(
                "validation_fraction", self.config.validation_fraction
            ),
            "early_stopping_rounds": self.hparams.get(
                "early_stopping_rounds", self.config.early_stopping_rounds
            ),
        }

        if self.task == "classification":
            return NGBClassifierCore(**common_kwargs)
        return NGBRegressorCore(**common_kwargs)

    def fit(self, x, y, X_val=None, y_val=None, sample_weight=None, val_sample_weight=None):
        self.estimator.fit(
            x,
            y,
            X_val=X_val,
            Y_val=y_val,
            sample_weight=sample_weight,
            val_sample_weight=val_sample_weight,
        )
        return self

    def predict(self, x):
        return self.estimator.predict(x)

    def predict_proba(self, x):
        if self.task != "classification":
            raise AttributeError("predict_proba is only available for classification models.")
        return self.estimator.predict_proba(x)

    def pred_dist(self, x, max_iter=None):
        return self.estimator.pred_dist(x, max_iter=max_iter)

    def raw_params(self, x):
        return self.estimator.pred_param(x)

    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_


class NGBSurvivalBackend(BoostingSurvivalBackendAdapter):
    """NGBoost survival backend adapter implementing the generic contract."""

    def __init__(self, config=DefaultNGBoostConfig(), **kwargs):
        self.config = config
        self.hparams = kwargs
        distribution = _resolve_distribution(
            self.hparams.get("distribution", self.config.distribution),
            default="survival",
        )
        score = _resolve_score(self.hparams.get("score", self.config.score))
        base_estimator = self.hparams.get("Base")
        if base_estimator is None:
            base_estimator = build_base_learner(
                kind=self.hparams.get("base_learner", self.config.base_learner),
                base_kwargs=self.hparams.get(
                    "base_learner_kwargs", self.config.base_learner_kwargs
                ),
            )

        self.estimator = NGBSurvivalCore(
            Dist=distribution,
            Score=score,
            Base=base_estimator,
            natural_gradient=self.hparams.get(
                "natural_gradient", self.config.natural_gradient
            ),
            n_estimators=self.hparams.get("n_estimators", self.config.n_estimators),
            learning_rate=self.hparams.get("learning_rate", self.config.learning_rate),
            minibatch_frac=self.hparams.get(
                "minibatch_frac", self.config.minibatch_frac
            ),
            col_sample=self.hparams.get("col_sample", self.config.col_sample),
            verbose=self.hparams.get("verbose", self.config.verbose),
            verbose_eval=self.hparams.get("verbose_eval", self.config.verbose_eval),
            tol=self.hparams.get("tol", self.config.tol),
            random_state=self.hparams.get("random_state", self.config.random_state),
        )

    def fit(self, x, t, e, X_val=None, T_val=None, E_val=None):
        self.estimator.fit(x, t, e, X_val=X_val, T_val=T_val, E_val=E_val)
        return self

    def predict(self, x):
        return self.estimator.predict(x)

    def pred_dist(self, x, max_iter=None):
        return self.estimator.pred_dist(x, max_iter=max_iter)

    def raw_params(self, x):
        return self.estimator.pred_param(x)

    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_


class NGBoost(GenericBoostingWrapper):
    """Generic boosting wrapper configured for NGBoost backends."""

    def __init__(self, **kwargs):
        super().__init__(backend=NGBoostBackend(**kwargs))
        self.estimator = self.backend.estimator


class NGBSurvival(GenericBoostingSurvivalWrapper):
    """Generic survival boosting wrapper configured for NGBoost."""

    def __init__(self, **kwargs):
        super().__init__(backend=NGBSurvivalBackend(**kwargs))
        self.estimator = self.backend.estimator
