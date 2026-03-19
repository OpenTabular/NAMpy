"""Internal NGBoost utilities adapted for NAMpy."""

from warnings import warn

import numpy as np
import scipy as sp
from scipy.stats import lognorm as lognorm_dist
from scipy.stats import norm as norm_dist
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_array, check_random_state, check_X_y


def y_from_censored(times, events=None):
    """Build the structured target array used by censored survival losses."""
    if times is None:
        return None

    if getattr(times, "dtype", None) == [("Event", "?"), ("Time", "<f8")]:
        return times

    times = check_array(times, ensure_2d=False)
    times = times.reshape(times.shape[0])

    if events is None:
        events = np.ones_like(times)
    else:
        events = check_array(events, ensure_2d=False)
        events = events.reshape(events.shape[0])

    y = np.empty(dtype=[("Event", np.bool_), ("Time", np.float64)], shape=times.shape[0])
    y["Event"] = events.astype(np.bool_)
    y["Time"] = times.astype(np.float64)
    return y


class Score:
    def total_score(self, y, sample_weight=None):
        return np.average(self.score(y), weights=sample_weight)

    def grad(self, y, natural=True):
        grad = self.d_score(y)
        if natural:
            metric = self.metric()
            grad = self._natural_gradient(grad, metric)
        return grad

    def _natural_gradient(self, grad, metric):
        grad = np.asarray(grad)
        metric = np.asarray(metric)

        if grad.ndim != 2:
            raise ValueError(f"Expected 2D gradients, got shape {grad.shape}.")
        if metric.ndim != 3:
            raise ValueError(f"Expected 3D metric tensor, got shape {metric.shape}.")
        if metric.shape[0] != grad.shape[0]:
            raise ValueError(
                f"Metric shape {metric.shape} is incompatible with gradient shape {grad.shape}."
            )
        if metric.shape[1:] != (grad.shape[1], grad.shape[1]):
            raise ValueError(
                f"Metric shape {metric.shape} is incompatible with gradient shape {grad.shape}."
            )

        try:
            return np.linalg.solve(metric, grad[..., None])[..., 0]
        except np.linalg.LinAlgError:
            result = np.zeros_like(grad)
            for idx in range(grad.shape[0]):
                try:
                    result[idx] = np.linalg.solve(metric[idx], grad[idx])
                except np.linalg.LinAlgError:
                    result[idx] = np.linalg.pinv(metric[idx]) @ grad[idx]
            return result


class LogScore(Score):
    def metric(self, n_mc_samples=100):
        grads = np.stack([self.d_score(y) for y in self.sample(n_mc_samples)])
        return np.mean(np.einsum("sik,sij->sijk", grads, grads), axis=0)


class CRPScore(Score):
    pass


class Distn:
    def __init__(self, params):
        self._params = params

    def __getitem__(self, key):
        return self.__class__(self._params[:, key])

    def __len__(self):
        return self._params.shape[1]

    @classmethod
    def implementation(cls, score_cls, scores=None):
        if scores is None:
            scores = cls.scores
        if score_cls in scores:
            warn(
                f"Using Dist={score_cls.__name__} is unnecessary. "
                "NGBoost automatically selects the correct implementation.",
                stacklevel=2,
            )
            return score_cls
        try:
            return {score.__bases__[-1]: score for score in scores}[score_cls]
        except KeyError as err:
            raise ValueError(
                f"The scoring rule {score_cls.__name__} is not implemented for "
                f"the {cls.__name__} distribution."
            ) from err

    @classmethod
    def uncensor(cls, score_cls):
        dist_score = cls.implementation(score_cls, cls.censored_scores)

        class UncensoredScore(dist_score, dist_score.__base__):
            def score(self, y):
                return super().score(y_from_censored(y))

            def d_score(self, y):
                return super().d_score(y_from_censored(y))

        class DistWithUncensoredScore(cls):
            scores = [UncensoredScore]

        return DistWithUncensoredScore


class RegressionDistn(Distn):
    def predict(self):
        return self.mean()


class ClassificationDistn(Distn):
    def predict(self):
        return np.argmax(self.class_probs(), axis=1)


def survival_distn_class(dist_cls):
    class SurvivalDistn(dist_cls):
        _basedist = dist_cls
        scores = dist_cls.censored_scores

        def fit(y):
            return dist_cls.fit(y["Time"])

    return SurvivalDistn


class NormalLogScore(LogScore):
    def score(self, y):
        return -self.dist.logpdf(y)

    def d_score(self, y):
        grad = np.zeros((len(y), 2))
        grad[:, 0] = (self.loc - y) / self.var
        grad[:, 1] = 1.0 - ((self.loc - y) ** 2) / self.var
        return grad

    def metric(self):
        fisher = np.zeros((self.var.shape[0], 2, 2))
        fisher[:, 0, 0] = 1.0 / self.var
        fisher[:, 1, 1] = 2.0
        return fisher


class Normal(RegressionDistn):
    n_params = 2
    scores = [NormalLogScore]

    def __init__(self, params):
        super().__init__(params)
        self.loc = params[0]
        self.scale = np.exp(params[1])
        self.var = self.scale**2
        self.dist = norm_dist(loc=self.loc, scale=self.scale)

    def fit(y):
        mean, scale = sp.stats.norm.fit(y)
        return np.array([mean, np.log(scale)])

    def sample(self, m):
        return np.array([self.rvs() for _ in range(m)])

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {"loc": self.loc, "scale": self.scale}


class LogNormalLogScoreCensored(LogScore):
    def score(self, y):
        event = y["Event"]
        time = y["Time"]
        censored = (1.0 - event) * np.log(1.0 - self.dist.cdf(time) + self.eps)
        uncensored = event * self.dist.logpdf(time)
        return -(censored + uncensored)

    def d_score(self, y):
        event = y["Event"][:, np.newaxis]
        time = y["Time"]
        log_time = np.log(time)
        z_score = (log_time - self.loc) / self.scale

        grad_uncensored = np.zeros((self.loc.shape[0], 2))
        grad_uncensored[:, 0] = (self.loc - log_time) / (self.scale**2)
        grad_uncensored[:, 1] = 1.0 - ((self.loc - log_time) ** 2) / (self.scale**2)

        grad_censored = np.zeros((self.loc.shape[0], 2))
        grad_censored[:, 0] = -sp.stats.norm.pdf(
            log_time, loc=self.loc, scale=self.scale
        ) / (1.0 - self.dist.cdf(time) + self.eps)
        grad_censored[:, 1] = (
            -z_score
            * sp.stats.norm.pdf(log_time, loc=self.loc, scale=self.scale)
            / (1.0 - self.dist.cdf(time) + self.eps)
        )

        return (1.0 - event) * grad_censored + event * grad_uncensored

    def metric(self):
        fisher = np.zeros((self.loc.shape[0], 2, 2))
        fisher[:, 0, 0] = 1.0 / (self.scale**2) + self.eps
        fisher[:, 1, 1] = 2.0
        return fisher


class LogNormal(RegressionDistn):
    n_params = 2
    censored_scores = [LogNormalLogScoreCensored]

    def __init__(self, params):
        super().__init__(params)
        self.loc = params[0]
        self.scale = np.exp(params[1])
        self.dist = lognorm_dist(s=self.scale, scale=np.exp(self.loc))
        self.eps = 1e-5

    def fit(y):
        mean, scale = sp.stats.norm.fit(np.log(y))
        return np.array([mean, np.log(scale)])

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {"s": self.scale, "scale": np.exp(self.loc)}


class PoissonLogScore(LogScore):
    def score(self, y):
        return -self.dist.logpmf(y)

    def d_score(self, y):
        grad = np.zeros((len(y), 1))
        grad[:, 0] = self.mu - y
        return grad

    def metric(self):
        fisher = np.zeros((self.mu.shape[0], 1, 1))
        fisher[:, 0, 0] = self.mu
        return fisher


class Poisson(RegressionDistn):
    n_params = 1
    scores = [PoissonLogScore]

    def __init__(self, params):
        super().__init__(params)
        self.logmu = params[0]
        self.mu = np.exp(self.logmu)
        self.dist = sp.stats.poisson(mu=self.mu)

    def fit(y):
        y = np.asarray(y)
        if not np.equal(np.mod(y, 1), 0).all():
            raise ValueError("All Poisson target data must be discrete integers.")
        if np.any(y < 0):
            raise ValueError("Poisson count data must be >= 0.")
        mean = max(float(np.mean(y)), 1e-12)
        return np.array([np.log(mean)])

    def sample(self, m):
        return np.array([self.dist.rvs() for _ in range(m)])

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {"mu": self.mu}


class GammaLogScore(LogScore):
    def score(self, y):
        return -self.dist.logpdf(y)

    def d_score(self, y):
        grad = np.zeros((len(y), 2))
        grad[:, 0] = self.alpha * (
            sp.special.digamma(self.alpha) - np.log(self.eps + self.beta * y)
        )
        grad[:, 1] = (self.beta * y) - self.alpha
        return grad

    def metric(self):
        fisher = np.zeros((self.alpha.shape[0], 2, 2))
        fisher[:, 0, 0] = self.alpha**2 * sp.special.polygamma(1, self.alpha)
        fisher[:, 1, 1] = self.alpha
        fisher[:, 0, 1] = -self.alpha
        fisher[:, 1, 0] = -self.alpha
        return fisher


class Gamma(RegressionDistn):
    n_params = 2
    scores = [GammaLogScore]

    def __init__(self, params):
        super().__init__(params)
        self.alpha = np.exp(params[0])
        self.beta = np.exp(params[1])
        self.dist = sp.stats.gamma(
            a=self.alpha, loc=np.zeros_like(self.alpha), scale=1.0 / self.beta
        )
        self.eps = 1e-10

    def fit(y):
        alpha, _, scale = sp.stats.gamma.fit(y, floc=0)
        return np.array([np.log(alpha), np.log(1.0 / scale)])

    def sample(self, m):
        return np.array([self.rvs() for _ in range(m)])

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {"alpha": self.alpha, "beta": self.beta}


class ExponentialLogScore(LogScore):
    def score(self, y):
        event, time = y["Event"], y["Time"]
        censored = (1 - event) * np.log(1 - self.dist.cdf(time) + 1e-10)
        uncensored = event * self.dist.logpdf(time)
        return -(censored + uncensored)

    def d_score(self, y):
        event, time = y["Event"], y["Time"]
        censored = (1 - event) * time.squeeze() / self.scale
        uncensored = event * (-1 + time.squeeze() / self.scale)
        return -(censored + uncensored).reshape((-1, 1))

    def metric(self):
        fisher = np.ones_like(self.scale)
        return fisher[:, np.newaxis, np.newaxis]


class Exponential(RegressionDistn):
    n_params = 1
    censored_scores = [ExponentialLogScore]

    def __init__(self, params):
        super().__init__(params)
        self.scale = np.exp(params[0])
        self.dist = sp.stats.expon(scale=self.scale)

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {"scale": self.scale}

    def fit(y):
        loc, scale = sp.stats.expon.fit(y)
        return np.array([np.log(loc + scale)])


class LaplaceLogScore(LogScore):
    def score(self, y):
        return -self.dist.logpdf(y)

    def d_score(self, y):
        grad = np.zeros((len(y), 2))
        grad[:, 0] = np.sign(self.loc - y) / self.scale
        grad[:, 1] = 1.0 - np.abs(self.loc - y) / self.scale
        return grad

    def metric(self):
        fisher = np.zeros((self.loc.shape[0], 2, 2))
        fisher[:, 0, 0] = 1.0 / self.scale**2
        fisher[:, 1, 1] = 1.0
        return fisher


class Laplace(RegressionDistn):
    n_params = 2
    scores = [LaplaceLogScore]

    def __init__(self, params):
        super().__init__(params)
        self.loc = params[0]
        self.logscale = params[1]
        self.scale = np.exp(params[1])
        self.dist = sp.stats.laplace(loc=self.loc, scale=self.scale)

    def fit(y):
        mean, scale = sp.stats.laplace.fit(y)
        return np.array([mean, np.log(scale)])

    def sample(self, m):
        return np.array([self.dist.rvs() for _ in range(m)])

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {"loc": self.loc, "scale": self.scale}


class CategoricalLogScore(LogScore):
    def score(self, y):
        return -np.log(self.probs[y, range(len(y))])

    def d_score(self, y):
        return (self.probs.T - np.eye(self.K_)[y])[:, 1 : self.K_]

    def metric(self):
        fisher = -np.einsum(
            "ji,ki->ijk", self.probs[1 : self.K_, :], self.probs[1 : self.K_, :]
        )
        diag = np.einsum("jii->ij", fisher)
        diag[:] += self.probs[1 : self.K_, :]
        return fisher


def k_categorical(num_classes):
    class Categorical(ClassificationDistn):
        scores = [CategoricalLogScore]
        problem_type = "classification"
        n_params = num_classes - 1
        K_ = num_classes

        def __init__(self, params):
            super().__init__(params)
            _, n_obs = params.shape
            self.logits = np.zeros((num_classes, n_obs))
            self.logits[1:num_classes, :] = params
            self.probs = sp.special.softmax(self.logits, axis=0)

        def fit(y):
            _, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return np.log(probs[1:num_classes]) - np.log(probs[0])

        def sample1(self):
            cumulative = np.cumsum(self.probs, axis=0)[:-1]
            interval = cumulative < np.random.random((1, len(self)))
            return np.sum(interval, axis=0)

        def sample(self, m):
            return np.array([self.sample1() for _ in range(m)])

        def class_probs(self):
            return self.probs.T

        @property
        def params(self):
            names = [f"p{idx}" for idx in range(self.n_params + 1)]
            return {name: prob for name, prob in zip(names, self.probs)}

    return Categorical


Bernoulli = k_categorical(2)


def manifold(score_cls, dist_cls):
    class Manifold(dist_cls.implementation(score_cls), dist_cls):
        pass

    return Manifold


def default_tree_learner():
    return DecisionTreeRegressor(
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        splitter="best",
        random_state=None,
    )


def default_linear_learner():
    return Ridge(alpha=0.0, random_state=None)


def build_base_learner(kind="tree", base_kwargs=None):
    base_kwargs = {} if base_kwargs is None else dict(base_kwargs)
    kind = str(kind).lower()

    if kind == "tree":
        learner = default_tree_learner()
    elif kind == "linear":
        learner = default_linear_learner()
    else:
        raise ValueError(
            f"Unsupported base learner {kind!r}. Use 'tree', 'linear', or pass an estimator."
        )

    learner.set_params(**base_kwargs)
    return learner


class NGBoostCore:
    def __init__(
        self,
        Dist=Normal,
        Score=LogScore,
        Base=None,
        natural_gradient=True,
        n_estimators=500,
        learning_rate=0.01,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
        validation_fraction=0.1,
        early_stopping_rounds=None,
    ):
        self.Dist = Dist
        self.Score = Score
        self.Base = default_tree_learner() if Base is None else Base
        self.Manifold = manifold(Score, Dist)
        self.natural_gradient = natural_gradient
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.col_sample = col_sample
        self.verbose = verbose
        self.verbose_eval = verbose_eval
        self.init_params = None
        self.n_features = None
        self.base_models = []
        self.scalings = []
        self.col_idxs = []
        self.tol = tol
        self.random_state = check_random_state(random_state)
        self.best_val_loss_itr = None
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.multi_output = bool(getattr(self.Dist, "multi_output", False))

    def fit_init_params_to_marginal(self, y, sample_weight=None, iters=1000):
        del sample_weight, iters
        self.init_params = self.Manifold.fit(y)

    def pred_param(self, x, max_iter=None):
        n_rows, _ = x.shape
        params = np.ones((n_rows, self.Manifold.n_params)) * self.init_params
        for idx, (models, scale, col_idx) in enumerate(
            zip(self.base_models, self.scalings, self.col_idxs)
        ):
            if max_iter is not None and idx == max_iter:
                break
            residuals = np.array([model.predict(x[:, col_idx]) for model in models]).T
            params -= self.learning_rate * residuals * scale
        return params

    def sample(self, x, y, sample_weight, params):
        row_idx = np.arange(len(y))
        col_idx = np.arange(x.shape[1])

        if self.minibatch_frac != 1.0:
            batch_size = int(self.minibatch_frac * len(y))
            row_idx = self.random_state.choice(np.arange(len(y)), batch_size, replace=False)

        if self.col_sample != 1.0:
            if self.col_sample > 0.0:
                col_size = max(1, int(self.col_sample * x.shape[1]))
            else:
                col_size = 0
            col_idx = self.random_state.choice(
                np.arange(x.shape[1]), col_size, replace=False
            )

        weight_batch = None if sample_weight is None else sample_weight[row_idx]
        return (
            row_idx,
            col_idx,
            x[row_idx, :][:, col_idx],
            y[row_idx],
            weight_batch,
            params[row_idx, :],
        )

    def fit_base(self, x, grads, sample_weight=None):
        if sample_weight is None:
            models = [clone(self.Base).fit(x, grad) for grad in grads.T]
        else:
            models = [
                clone(self.Base).fit(x, grad, sample_weight=sample_weight)
                for grad in grads.T
            ]
        fitted = np.array([model.predict(x) for model in models]).T
        self.base_models.append(models)
        return fitted

    def line_search(self, residuals, start, y, sample_weight=None, scale_init=1.0):
        dist_init = self.Manifold(start.T)
        loss_init = dist_init.total_score(y, sample_weight)
        scale = scale_init

        while True:
            scaled = residuals * scale
            dist = self.Manifold((start - scaled).T)
            loss = dist.total_score(y, sample_weight)
            if not np.isfinite(loss) or loss > loss_init or scale > 256:
                break
            scale *= 2.0

        while True:
            scaled = residuals * scale
            dist = self.Manifold((start - scaled).T)
            loss = dist.total_score(y, sample_weight)
            norm = np.mean(np.linalg.norm(scaled, axis=1))
            if norm < self.tol:
                break
            if np.isfinite(loss) and loss < loss_init:
                break
            scale *= 0.5

        self.scalings.append(scale)
        return scale

    def fit(
        self,
        x,
        y,
        X_val=None,
        Y_val=None,
        sample_weight=None,
        val_sample_weight=None,
        train_loss_monitor=None,
        val_loss_monitor=None,
        early_stopping_rounds=None,
    ):
        self.base_models = []
        self.scalings = []
        self.col_idxs = []

        return self.partial_fit(
            x,
            y,
            X_val=X_val,
            Y_val=Y_val,
            sample_weight=sample_weight,
            val_sample_weight=val_sample_weight,
            train_loss_monitor=train_loss_monitor,
            val_loss_monitor=val_loss_monitor,
            early_stopping_rounds=early_stopping_rounds,
        )

    def partial_fit(
        self,
        x,
        y,
        X_val=None,
        Y_val=None,
        sample_weight=None,
        val_sample_weight=None,
        train_loss_monitor=None,
        val_loss_monitor=None,
        early_stopping_rounds=None,
    ):
        if len(self.base_models) != len(self.scalings) or len(self.base_models) != len(
            self.col_idxs
        ):
            raise RuntimeError("Base models, scalings, and col_idxs are not the same length.")

        if self.early_stopping_rounds is not None:
            early_stopping_rounds = self.early_stopping_rounds
            if X_val is None and Y_val is None:
                if self.verbose:
                    print(
                        "early_stopping_rounds is set but no validation set was passed; "
                        f"creating one with validation_fraction={self.validation_fraction}."
                    )
                if sample_weight is None:
                    x, X_val, y, Y_val = train_test_split(
                        x,
                        y,
                        test_size=self.validation_fraction,
                        random_state=self.random_state,
                    )
                else:
                    (
                        x,
                        X_val,
                        y,
                        Y_val,
                        sample_weight,
                        val_sample_weight,
                    ) = train_test_split(
                        x,
                        y,
                        sample_weight,
                        test_size=self.validation_fraction,
                        random_state=self.random_state,
                    )
            elif (X_val is None) ^ (Y_val is None):
                raise ValueError("Inconsistent validation data: both X_val and Y_val are required.")

        if y is None:
            raise ValueError("y cannot be None")

        x, y = check_X_y(
            x,
            y,
            accept_sparse=True,
            ensure_all_finite="allow-nan",
            multi_output=self.multi_output,
            y_numeric=True,
        )

        self.n_features = x.shape[1]
        self.fit_init_params_to_marginal(y)
        params = self.pred_param(x)
        train_losses = []

        if X_val is not None and Y_val is not None:
            X_val, Y_val = check_X_y(
                X_val,
                Y_val,
                accept_sparse=True,
                ensure_all_finite="allow-nan",
                multi_output=self.multi_output,
                y_numeric=True,
            )
            val_params = self.pred_param(X_val)
            val_losses = []
            best_val_loss = np.inf
        else:
            val_losses = None

        if train_loss_monitor is None:
            train_loss_monitor = lambda dist, y_batch, weights: dist.total_score(  # noqa: E731
                y_batch, sample_weight=weights
            )
        if val_loss_monitor is None:
            val_loss_monitor = lambda dist, y_batch: dist.total_score(  # noqa: E731
                y_batch, sample_weight=val_sample_weight
            )

        for itr in range(len(self.col_idxs), self.n_estimators + len(self.col_idxs)):
            _, col_idx, x_batch, y_batch, weight_batch, p_batch = self.sample(
                x, y, sample_weight, params
            )
            self.col_idxs.append(col_idx)

            dist = self.Manifold(p_batch.T)
            loss = train_loss_monitor(dist, y_batch, weight_batch)
            train_losses.append(loss)
            grads = dist.grad(y_batch, natural=self.natural_gradient)

            projected_grad = self.fit_base(x_batch, grads, weight_batch)
            scale = self.line_search(projected_grad, p_batch, y_batch, weight_batch)

            params -= (
                self.learning_rate
                * scale
                * np.array([model.predict(x[:, col_idx]) for model in self.base_models[-1]]).T
            )

            val_loss = 0.0
            if X_val is not None and Y_val is not None:
                val_params -= (
                    self.learning_rate
                    * scale
                    * np.array(
                        [model.predict(X_val[:, col_idx]) for model in self.base_models[-1]]
                    ).T
                )
                val_loss = val_loss_monitor(self.Manifold(val_params.T), Y_val)
                val_losses.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_val_loss_itr = itr
                if (
                    early_stopping_rounds is not None
                    and len(val_losses) > early_stopping_rounds
                    and best_val_loss < np.min(np.asarray(val_losses[-early_stopping_rounds:]))
                ):
                    if self.verbose:
                        print("== Early stopping achieved.")
                    break

            if self.verbose and int(self.verbose_eval) > 0 and itr % int(self.verbose_eval) == 0:
                grad_norm = np.linalg.norm(grads, axis=1).mean() * scale
                print(
                    f"[iter {itr}] loss={loss:.4f} val_loss={val_loss:.4f} "
                    f"scale={scale:.4f} norm={grad_norm:.4f}"
                )

            if np.linalg.norm(projected_grad, axis=1).mean() < self.tol:
                if self.verbose:
                    print(f"== Quitting at iteration {itr} due to gradient norm.")
                break

        metric_name = self.Score.__name__.upper()
        self.evals_result = {"train": {metric_name: train_losses}}
        if val_losses is not None:
            self.evals_result["val"] = {metric_name: val_losses}
        return self

    def pred_dist(self, x, max_iter=None):
        x = check_array(x, accept_sparse=True, ensure_all_finite="allow-nan")
        params = np.asarray(self.pred_param(x, max_iter))
        return self.Dist(params.T)

    def staged_pred_dist(self, x, max_iter=None):
        predictions = []
        n_rows, _ = x.shape
        params = np.ones((n_rows, self.Dist.n_params)) * self.init_params
        for idx, (models, scale, col_idx) in enumerate(
            zip(self.base_models, self.scalings, self.col_idxs), start=1
        ):
            residuals = np.array([model.predict(x[:, col_idx]) for model in models]).T
            params -= self.learning_rate * residuals * scale
            predictions.append(self.Dist(np.copy(params.T)))
            if max_iter is not None and idx == max_iter:
                break
        return predictions

    def predict(self, x, max_iter=None):
        x = check_array(x, accept_sparse=True, ensure_all_finite="allow-nan")
        return self.pred_dist(x, max_iter=max_iter).predict()

    @property
    def feature_importances_(self):
        if not self.base_models:
            return None
        if not isinstance(self.base_models[0][0], DecisionTreeRegressor):
            return None

        params_trees = zip(*self.base_models)
        all_importances = [
            [
                self._get_feature_importance(tree, tree_index)
                for tree_index, tree in enumerate(trees)
            ]
            for trees in params_trees
        ]

        if not all_importances:
            return np.zeros(
                (len(self.base_models[0]), self.base_models[0][0].n_features_),
                dtype=np.float64,
            )

        all_importances = np.average(all_importances, axis=1, weights=self.scalings)
        denom = np.sum(all_importances, axis=1, keepdims=True)
        denom[denom == 0.0] = 1.0
        return all_importances / denom

    def _get_feature_importance(self, tree, tree_index):
        importances = np.zeros(self.n_features)
        importances[self.col_idxs[tree_index]] = getattr(tree, "feature_importances_")
        return importances


class NGBRegressorCore(NGBoostCore):
    def __init__(self, Dist=Normal, **kwargs):
        if not issubclass(Dist, RegressionDistn):
            raise ValueError(f"{Dist.__name__} is not usable for regression.")
        if not hasattr(Dist, "scores"):
            Dist = Dist.uncensor(LogScore)
        super().__init__(Dist=Dist, **kwargs)
        self._estimator_type = "regressor"


class NGBClassifierCore(NGBoostCore):
    def __init__(self, Dist=Bernoulli, **kwargs):
        if not issubclass(Dist, ClassificationDistn):
            raise ValueError(f"{Dist.__name__} is not usable for classification.")
        super().__init__(Dist=Dist, **kwargs)
        self._estimator_type = "classifier"

    def predict_proba(self, x, max_iter=None):
        return self.pred_dist(x, max_iter=max_iter).class_probs()


class NGBSurvivalCore(NGBoostCore):
    def __init__(self, Dist=LogNormal, **kwargs):
        if not issubclass(Dist, RegressionDistn):
            raise ValueError(f"{Dist.__name__} is not usable for survival.")
        if not hasattr(Dist, "censored_scores"):
            raise ValueError(
                f"The {Dist.__name__} distribution does not have censored scores implemented."
            )
        super().__init__(Dist=survival_distn_class(Dist), **kwargs)

    def fit(self, x, t, e, X_val=None, T_val=None, E_val=None, **kwargs):
        x = check_array(x, accept_sparse=True)
        if X_val is not None:
            X_val = check_array(X_val, accept_sparse=True)
        return super().fit(
            x,
            y_from_censored(t, e),
            X_val=X_val,
            Y_val=y_from_censored(T_val, E_val),
            **kwargs,
        )


SUPPORTED_DISTRIBUTIONS = {
    "exponential": Exponential,
    "gamma": Gamma,
    "laplace": Laplace,
    "normal": Normal,
    "lognormal": LogNormal,
    "poisson": Poisson,
    "bernoulli": Bernoulli,
}
