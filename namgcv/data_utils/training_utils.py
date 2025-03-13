import copy
import pickle
from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

import optax

from flax.training.train_state import TrainState
from flax.linen import Module

from mile.inference.metrics import RegressionMetrics
from mile.types import ParamTree


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
                    v[gen_idx].reshape(-1, 1).T
                    for gen_idx in next(index_generator(num_features, batch_size))
                ], axis=0
            ) for k, v in num_features.items()
        } if num_features is not None else {},
        "cat_features": {
            k: jnp.concatenate(
                arrays=[
                    v[gen_idx].reshape(-1, 1).T
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


def save_tree(
        dir: str | Path,
        tree: ParamTree
):
    """Save tree in .pkl format."""
    with open(dir / 'tree', 'wb') as f:
        pickle.dump(tree, f)


def load_tree(
        dir: str | Path
) -> ParamTree:
    """Load tree in .pkl format."""
    with open(dir / 'tree', 'rb') as f:
        return pickle.load(f)


def get_flattened_keys(
        d: dict,
        sep='.'
) -> list[str]:
    """Recursively get `sep` delimited path to the leaves of a tree.

    Parameters:
    -----------
    d: dict
        Parameter Tree to get the names of the leaves from.
    sep: str
        Separator for the tree path.

    Returns:
    --------
        list of names of the leaves in the tree.
    """
    keys = []
    for k, v in d.items():
        if isinstance(v, dict):
            keys.extend([f'{k}{sep}{kk}' for kk in get_flattened_keys(v)])
        else:
            keys.append(k)
    return keys


def save_params(
        dir: str | Path,
        params: ParamTree,
        idx: int | None = None
):
    """
    Save model parameters to disk.

    Args:
        dir: str | Path - directory to save the parameters to
        params: dict - parameters to save
        idx: int | None - index to append to the file name (default: None)
    """
    if not isinstance(dir, Path):
        dir = Path(dir)

    if not dir.exists():
        dir.mkdir(parents=True)

    leaves, tree = jax.tree.flatten(params)
    if not (dir.parent / 'tree').exists():
        save_tree(dir.parent, tree)

    param_names = get_flattened_keys(params)
    name = f'params_{idx}.npz' if idx is not None else 'params.npz'

    np.savez_compressed(dir / name, **dict(zip(param_names, leaves)))


import copy
import numpy as np


def _flatten_flax_dict(flax_params_dict):
    """
    Helper to flatten one "Flax-style" dict into a NumPyro-style dict.

    For example, a nested structure like:
        {
          "subnetwork_categorical_x": {
            "Dense": {
              "kernel": ...,
              "bias": ...
            }
          },
          "bias": ...
        }
    becomes something like:
        {
          "categorical_x/dense_kernel": ...,
          "categorical_x/dense_bias": ...,
          "bias": ...
        }
    """
    flat_params = {}
    for top_name, top_params in flax_params_dict.items():
        if "subnetwork" in top_name:
            # e.g. top_name = "subnetwork_categorical_x"
            # Identify the feature_name portion by stripping out the first two underscores
            prefixes = top_name.split("_")[:2]  # ["subnetwork", "categorical"]
            feature_name = copy.deepcopy(top_name)
            for prefix in prefixes:
                feature_name = feature_name.replace(f"{prefix}_", "")

            for layer_name, layer_dict in top_params.items():
                # Convert CamelCase to snake_case
                snake_layer_name = "".join(
                    ("_" + c.lower() if c.isupper() else c) for c in layer_name
                ).lstrip("_")

                for param_name, param_value in layer_dict.items():
                    new_key = f"{feature_name}/{snake_layer_name}_{param_name}"
                    flat_params[new_key] = param_value
        else:
            # e.g. a global "bias" or "intercept"
            flat_params[top_name] = top_params
    return flat_params


def map_flax_to_numpyro(flax_params_list: list[dict], expected_chains: int):
    """
    Convert a list of Flax parameter dicts (in certain patterns) into
    a NumPyro-style dict with leading dimension = `expected_chains`.

    Cases:
      1) A list of length k (>1) with k single-chain dicts:
         - Flatten each dict
         - Stack to shape (k, ...)
         - Must have k == expected_chains

      2) A list of length 1 with exactly one dict:
         - If expected_chains == 1,
            interpret as single-chain; flatten and add leading dim.
         - If expected_chains > 1,
            interpret that the single dict already has a leading dimension k == expected_chains;
            flatten and return as is (after shape check).

    All other inputs raise ValueError.
    """
    # Must be a list
    if not isinstance(flax_params_list, list):
        raise ValueError("Expected a list of dict(s). Got type: {}".format(type(flax_params_list)))

    # Case (1): multiple dicts, each single-chain => stack
    if len(flax_params_list) > 1:
        k = len(flax_params_list)
        if k != expected_chains:
            raise ValueError(
                f"Received {k} dicts in the list but `expected_chains={expected_chains}`."
            )

        # Flatten each dict
        flat_dicts = []
        for i, d in enumerate(flax_params_list):
            if not isinstance(d, dict):
                raise ValueError(f"Element at index {i} is not a dict.")
            flat_dicts.append(_flatten_flax_dict(d))

        # Check that all flattened dicts have identical keys
        param_names_per_chain = [set(d.keys()) for d in flat_dicts]
        common_keys = set.intersection(*param_names_per_chain)
        if any(names != common_keys for names in param_names_per_chain):
            raise ValueError(
                "All chain dicts must have identical parameter names to be stackable."
            )

        # Stack each parameter along axis=0 to shape (k, ...).
        stacked_params = {}
        for key in sorted(common_keys):
            arrays = [fd[key] for fd in flat_dicts]
            stacked_params[key] = np.stack(arrays, axis=0)

        return stacked_params

    # Case (2): list of length 1 => either single chain or multi-chain.
    elif len(flax_params_list) == 1:
        single_item = flax_params_list[0]
        if not isinstance(single_item, dict):
            raise ValueError("The single list element must be a dict.")

        # Flatten it
        flat_dict = _flatten_flax_dict(single_item)

        if expected_chains == 1:
            # Single-chain scenario.
            # Add a leading dimension of size 1 to each parameter.
            return {key: np.expand_dims(arr, axis=0) for key, arr in flat_dict.items()}
        else:
            # Multi-chain scenario =>
            # assume all parameters have leading dimension = expected_chains.
            for key, arr in flat_dict.items():
                if len(arr.shape) == 0:
                    raise ValueError(
                        f"Parameter '{key}' has scalar shape {arr.shape}; "
                        f"cannot interpret as {expected_chains} chains."
                    )
                if arr.shape[0] != expected_chains:
                    raise ValueError(
                        f"Parameter '{key}' has leading dimension {arr.shape[0]}, "
                        f"but expected {expected_chains}."
                    )
            return flat_dict  # Return as-is.

    else: # e.g. empty list
        raise ValueError(
            "Expected a list of length >= 1. "
            "Either multiple dicts or a single dict in the list."
        )


def merge_data_dicts(dict_list):
    """
    Recursively merge a list of dictionaries into a single dictionary.
    At the lowest level, the arrays are concatenated row-wise (axis=0).

    Parameters
    ----------
    dict_list : list[dict]
        List of dictionaries with identical nested keys.

    Returns
    -------
    dict
        A single dictionary with the same nested structure as the inputs, where
        the arrays at the deepest level have been concatenated.
    """
    if not dict_list:
        return {}

    merged = {}
    for key in dict_list[0]:
        values = [d[key] for d in dict_list]
        if isinstance(values[0], dict):
            merged[key] = merge_data_dicts(values)
        else:
            merged[key] = jnp.concatenate(values, axis=0)
    return merged