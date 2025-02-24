from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp

import optax

from flax.training.train_state import TrainState
from flax.linen import Module

from mile.inference.metrics import RegressionMetrics


def get_single_input(
        num_features: Dict[str, jnp.ndarray] = None,
        cat_features: Dict[str, jnp.ndarray] = None,
        batch_size: int = 1
):
    """
    Helper function which returns a random input data of shape (batch_size, ...).
    Note: the generator is discarded after the first call, so this method cannot be used as a
    dataset iterator.

    Parameters
    ----------
    num_features : dict
        Dictionary of numerical features with feature names as keys.
    cat_features:
        Dictionary of categorical features with feature names as keys.
    batch_size : int
        The desired batch size for the input data.

    Returns
    -------
    jnp.ndarray
        A random input data of shape (batch_size, ...).
    """

    def index_generator(
            feature_dict: Dict[str, jnp.ndarray],
            batch_size: int = 1
    ):
        """
        Generator function that yields batches of indices 0, 1, ..., n-1,
        where n is the length of the first array in the feature dictionary.

        Parameters
        ----------
        feature_dict:
            The feature dictionary.
        batch_size:
            The batch size.

        Returns
        -------
        Generator
            A generator yielding the indices.
        """

        first_array = next(iter(feature_dict.values()))
        n = len(first_array)
        for i in range(0, n, batch_size):
            yield list(range(i, min(i + batch_size, n)))

    return {
        "num_features": {
            k: jnp.concatenate(
                arrays=[
                    v[gen_idx].reshape(-1, 1)
                    for gen_idx in next(index_generator(num_features, batch_size))
                ], axis=0
            ) for k, v in num_features.items()
        } if num_features is not None else {},
        "cat_features": {
            k: jnp.concatenate(
                arrays=[
                    v[gen_idx].reshape(-1, 1)
                    for gen_idx in next(index_generator(cat_features, batch_size))
                ], axis=0
            ) for k, v in cat_features.items()
        } if cat_features is not None else {}
    }


def get_initial_state(
        rng: jnp.ndarray,
        x: jnp.ndarray,
        module: Module,
        optimizer: optax.GradientTransformation,
) -> TrainState:
    """
    Get the initial flax modeule training state.

    Parameters
    ----------
    rng : jnp.ndarray
        Random number generator key.
    x : jnp.ndarray
        Input data.
    module : nn.Module
        Flax module.
    optimizer : optax.GradientTransformation
        Optimizer.

    Returns
    -------
    TrainState:
        The initial training state.
    """

    rng, dropout_rng = jax.random.split(rng)
    rng, batch_norm_rng = jax.random.split(rng)
    rng, layer_norm_rng = jax.random.split(rng)
    rng, params_rng = jax.random.split(rng)
    params = module.init(
        rngs={
            "params": params_rng,
            "dropout": dropout_rng,
            "batch_norm": batch_norm_rng,
            "layer_norm": layer_norm_rng
        },
        num_features=x["num_features"],
        cat_features=x["cat_features"],
        train=True
    )["params"]

    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=optimizer
    )


def gaussian_nll_loss(
        x: jnp.ndarray,
        mu: jnp.ndarray,
        sigma: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute the Gaussian negative log-likelihood loss.

    Parameters:
    ----------
    x : jnp.array
        The target values of shape (n_obs).
    mu : jnp.array
        The predicted mean values of shape (..., n_obs).
    sigma : jnp.array
        The predicted standard deviation values of shape (..., n_obs).

    Returns:
    ----------
    jnp.array:
        The computed loss.
    """
    sigma = jnp.clip(sigma, 1e-5)  # Ensure stability in calculations
    return 0.5 * jnp.log(2 * jnp.pi * sigma ** 2) + ((x - mu) ** 2) / (2 * sigma ** 2)

def single_train_step_wrapper(
        state: TrainState,
        batch: dict[str, jnp.ndarray],
        early_stop: bool = False,
        rng: jnp.ndarray = None,
) -> TrainState:
    """
    Perform a single training step.

    Parameters
    ----------
    rng:
        Random number generator.
    state: TrainState
        The current training state.
    batch:
        The batch of data.
    early_stop: bool
        Early stopping condition.

    Returns
    -------

    """

    def loss_fn(params: dict) -> tuple[jax.Array, RegressionMetrics]:
        """
        Compute the loss function for the model.

        Parameters
        ----------
        params: dict
            The parameters of the model.

        Returns
        -------
        tuple[jax.Array, RegressionMetrics]:
            The computed loss and the metrics object.
        """
        train = True
        logits = state.apply_fn(
            {'params': params},
            num_features=batch["feature"]["numerical"],
            cat_features=batch["feature"]["categorical"],
            train=train,
            rng=dropout_rng
        )
        loss = gaussian_nll_loss(
            x=batch["target"],
            mu=logits[..., 0],
            sigma=logits[..., 1].clip(min=1e-6, max=1e6),
        )

        metrics = compute_regression_metrics(
            logits=logits,
            y=batch["target"],
            step=state.step
        )

        return loss.mean(), metrics

    def _single_step(state: TrainState):
        """
        Single training step function.

        Parameters
        ----------
        state: TrainState
            The current training state.

        Returns
        -------
        (TrainState, RegressionMetrics):
            The updated training state and the metrics object.
        """
        grad_fn = jax.value_and_grad(
            loss_fn,
            has_aux=True
        )
        (_, metrics), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)

        return state, metrics

    def _fallback(state: TrainState):
        """
        Fallback function for the training step.

        Parameters
        ----------
        state: TrainState
            The current training state.

        Returns
        -------
        (TrainState, RegressionMetrics):
            The updated training state and the metrics object.
        """

        return (
            state,
            RegressionMetrics(
                step=state.step,
                nlll=jnp.nan,
                rmse=jnp.nan
            )
        )

    rng, dropout_rng = jax.random.split(rng)

    return jax.lax.cond(
        early_stop,
        _fallback,
        _single_step,
        state, # args to _single_step.
    )


def compute_regression_metrics(
        logits: jnp.ndarray,
        y: jnp.ndarray,
        step: jnp.ndarray = jnp.nan,
):
    """
    Compute the metrics for regression Task.

    Parameters
    ----------
        logits: jnp.ndarray
            Logits array.
        y: jnp.ndarray
            Target array.
        step: jnp.ndarray
            Step array, used for tracking steps in the pipeline.

    Returns
    -------
        RegressionMetrics:
            Metrics object containing recorded metrics.
    """

    loss = gaussian_nll_loss(
        x=y,
        mu=logits[..., 0],
        sigma=jnp.exp(logits[..., 1]).clip(min=1e-6, max=1e6),
    )
    se = (y - logits[..., 0]) ** 2
    metrics = RegressionMetrics(
        step=step,
        nlll=loss.mean(),
        rmse=jnp.sqrt(se.mean())
    )
    return metrics


def single_prediction_wrapper(
    state: TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    early_stop: bool = False,
    train: bool = True
) -> RegressionMetrics:
    """
    Predict the model for a regression task.

    Parameters
    ----------
        state: TrainState
            Training State.
        x: jnp.ndarray
            Input data of shape (B, ...).
        y: jnp.ndarray
            Target data of shape (B,).
        early_stop: bool
            Early stopping condition used in pipeline.

    Returns
    -------
        (RegressionMetrics): Regression Metrics object containing the metrics.
    """

    def _pred(
            state: TrainState,
            x: jnp.ndarray,
            y: jnp.ndarray,
    ):
        logits = state.apply_fn(
            {'params': state.params},
            num_features=x["numerical"],
            cat_features=x["categorical"],
            train=train
        )
        return compute_regression_metrics(logits, y, step=state.step)

    def _fallback(*args, **kwargs):
        metrics = RegressionMetrics(step=state.step, nlll=jnp.nan, rmse=jnp.nan)
        return metrics

    return jax.lax.cond(early_stop, _fallback, _pred, state, x, y)


def early_stop_check(
        losses: jnp.ndarray,
        patience: int
):
    """
    Check early stopping condition for losses shape (n_devices, N_steps).

    Parameters
    ----------
        losses: jnp.ndarray
            Losses array.
        patience: int
            Number of Patience steps for early stopping.

    Returns
    -------
        jnp.ndarray:
            Boolean array of shape (n_devices,) where True indicates to stop.
    """

    if losses.shape[-1] < patience:
        return jnp.repeat(False, len(losses))

    reference_loss = losses[:, -(patience + 1)]
    reference_loss = jnp.expand_dims(reference_loss, axis=-1)
    recent_losses = losses[:, -(patience):]

    return jnp.all(recent_losses >= reference_loss, axis=1)
