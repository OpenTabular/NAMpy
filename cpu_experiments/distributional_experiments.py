import os

import scipy.stats

num_chains = 10
n_devices = min(os.cpu_count(), num_chains)
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={n_devices}'
import pickle

import pandas as pd
import numpy as np
import jax.numpy as jnp

import numpyro

import matplotlib.pyplot as plt
import seaborn as sns

import jax
from jax.lib import xla_bridge

print(xla_bridge.get_backend().platform)
print(f"Default backend for JAX: {jax.default_backend()}")
print(
    f"Number of devices available on default backend: "
    f"{jax.local_device_count(backend=jax.default_backend())}"
)

from namgcv.basemodels.bnam import BayesianNAM, link_location

from numpyro import handlers
from numpyro import distributions as dist

from jax import random
from scipy import stats

import numpy as np
import pandas as pd
import numpyro
from numpyro import handlers, distributions as dist
import jax.random as random


def get_invgamma_synthetic_data(n_samples: int=3000, seed=42, plot=True):
    """
    Function to generate synthetic data for testing the BayesianNAM model.
    Returns a pandas DataFrame containing the synthetic data.

    Args:
        n_samples (int): Number of samples to generate.

    Returns:
        pd.DataFrame: DataFrame containing the synthetic data.
    """

    lower_bound, upper_bound = 0, 3
    n_features = 2
    X = np.random.uniform(lower_bound, upper_bound, (n_samples, n_features))
    x1, x2 = (
        X[:, 0],
        X[:, 1]
    )

    def theta1_func(x1, x2):
        return (
                # 2
                # +
                # np.sin(x1)
                # +
                # 0.5 * x2

                np.log1p(np.exp(
                    # np.sin(2.5*x1)
                    # + 0.3 * (x2 - 1.5) ** 2
                    # + 8

                    np.tanh(1.5 * x1 - 2)
                    + 0.4 * np.sqrt(x2 + 0.1)
                ))
        )

    def theta2_func(x1, x2):
        return (
                # 1
                # +
                # np.cos(x2)
                # +
                # 0.3 * x1**2
                np.log1p(np.exp(
                    # np.cos(1.5*x2)
                    # + 0.5 * np.log(1 + x1 ** 2)
                    # - 0.5

                    -0.3 * (x1 - 1.5) ** 2
                    + np.exp(-0.5 * x2)
                    + 0.5
                ))
        )

    theta_functions = [
        theta1_func,
        theta2_func,
    ]

    theta1 = theta_functions[0](x1, x2)
    theta2 = theta_functions[1](x1, x2)

    def model():
        with numpyro.plate(name="data", size=theta1.shape[0]):
            y = numpyro.sample(
                name="y",
                fn=dist.InverseGamma(theta1, theta2),
                rng_key=random.PRNGKey(42)
            )

    model_trace = handlers.trace(model).get_trace()
    y = model_trace["y"]["value"]

    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        sns.histplot(theta1, bins=50, kde=True, ax=axs[0], color='skyblue')
        axs[0].set_title('θ1')
        axs[0].set_xlabel('θ1')
        axs[0].grid(True)
        sns.histplot(theta2, bins=50, kde=True, ax=axs[1], color='salmon')
        axs[1].set_title('θ2')
        axs[1].set_xlabel('θ2')
        axs[1].grid(True)
        sns.histplot(y, bins=50, kde=True, ax=axs[2], color='limegreen')
        axs[2].set_title('Y ~ InvGamma(θ1, θ2)')
        axs[2].set_xlabel('Y')
        axs[2].grid(True)
        plt.tight_layout()
        plt.show()

        # Visualize marginal feature effects.
        x_grid = np.linspace(lower_bound, upper_bound, 100)
        theta_labels = ["θ1", "θ2"]
        feature_labels = ["x1", "x2"]

        sns.set_style("white", {"axes.grid": True})
        sns.set_palette("Reds")
        fig, axes = plt.subplots(
            n_features,
            len(theta_functions),
            figsize=(6*(len(theta_functions)+1), 6*(n_features+1))
        )
        for i in range(n_features):  # for each feature (x1, x2)
            for j, theta_func in enumerate(theta_functions):  # for each theta (θ1, θ2)
                pdp_vals = []
                for val in x_grid:
                    X_temp = X.copy()
                    X_temp[:, i] = val  # Fix the i-th feature.
                    y_vals = theta_func(X_temp[:, 0], X_temp[:, 1])
                    pdp_vals.append(np.mean(y_vals))

                feature_data = X[:, i]
                theta_data = theta_func(X[:, 0], X[:, 1])

                ax = axes[i, j]
                sns.scatterplot(x=feature_data, y=theta_data, alpha=0.5, ax=ax)
                ax.plot(x_grid, pdp_vals, color="black", linewidth=2)

                if i == 0:
                    ax.set_title(f'{theta_labels[j]}', fontsize=12)
                if j == 0:
                    ax.set_ylabel(f'{feature_labels[i]}', fontsize=12)
                ax.set_xlabel(feature_labels[i])
                ax.grid(True)

        plt.tight_layout()
        plt.show()

    return pd.DataFrame(
        data={
            'x1': x1,
            'x2': x2,
            'theta1': theta1,
            'theta2': theta2,
            'response': y,
        }
    )


def get_gaussian_synthetic_data(n_samples: int=3000, seed=42, plot:bool = True):
    lower_bound, upper_bound = -1, 1
    n_features = 2
    X = np.random.uniform(lower_bound, upper_bound, (n_samples, n_features))
    x1, x2 = X[:, 0], X[:, 1]

    def theta1_func(x1, x2):
        return (
            # 2 + np.sin(1.2 * x1) + 0.5 * np.log1p(x2**2)
            x1**2 + np.sin(4*x2)
        )

    def theta2_func(x1, x2):
        # z = 0.3 * (x1 - 1.5)**2 + np.exp(-0.5 * x2)
        z = 0.01 + np.exp(x1)/10 + 0.1 + x2**2
        # return np.log1p(np.exp(z))
        return z

    theta_functions = [theta1_func, theta2_func]

    theta1 = theta1_func(x1, x2)
    theta2 = theta2_func(x1, x2)

    # Sample from Gaussian
    y = np.random.normal(loc=theta1, scale=theta2)

    if plot:
        # Plot distributions
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        sns.histplot(theta1, bins=50, kde=True, ax=axs[0], color='skyblue')
        axs[0].set_title('θ1')
        axs[0].set_xlabel('θ2')
        axs[0].grid(True)

        sns.histplot(theta2, bins=50, kde=True, ax=axs[1], color='salmon')
        axs[1].set_title('θ2')
        axs[1].set_xlabel('θ2')
        axs[1].grid(True)

        sns.histplot(y, bins=50, kde=True, ax=axs[2], color='limegreen')
        axs[2].set_title('Y ~ Normal(θ1, θ2)')
        axs[2].set_xlabel('Y')
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()

        # Partial dependence plots
        x_grid = np.linspace(lower_bound, upper_bound, 100)
        theta_labels = ["θ1", "θ2"]
        feature_labels = ["x1", "x2"]

        sns.set_style("white", {"axes.grid": True})
        sns.set_palette("Reds")
        fig, axes = plt.subplots(
            n_features,
            len(theta_functions),
            figsize=(6 * len(theta_functions + 1), 6 * (n_features + 1))
        )

        for i in range(n_features):
            for j, theta_func in enumerate(theta_functions):
                pdp_vals = []
                for val in x_grid:
                    X_temp = X.copy()
                    X_temp[:, i] = val
                    y_vals = theta_func(X_temp[:, 0], X_temp[:, 1])
                    pdp_vals.append(np.mean(y_vals))

                feature_data = X[:, i]
                theta_data = theta_func(X[:, 0], X[:, 1])

                ax = axes[i, j]
                sns.scatterplot(x=feature_data, y=theta_data, alpha=0.5, ax=ax)
                ax.plot(x_grid, pdp_vals, color="black", linewidth=2)

                if i == 0:
                    ax.set_title(f'{theta_labels[j]}', fontsize=12)
                if j == 0:
                    ax.set_ylabel(f'{feature_labels[i]}', fontsize=12)
                ax.set_xlabel(feature_labels[i])
                ax.grid(True)

        plt.tight_layout()
        plt.show()

    return pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'theta1': theta1,
        'theta2': theta2,
        'response': y,
    })


def get_beta_synthetic_data(n_samples: int=3000, seed=42, plot:bool = True):
    lower_bound, upper_bound = 0, 3
    n_features = 2
    X = np.random.uniform(lower_bound, upper_bound, (n_samples, n_features))
    x1, x2 = (
        X[:, 0],
        X[:, 1]
    )

    def theta1_func(x1, x2):
        return (
            np.log1p(np.exp(
                np.sin(1.2 * x1) + 0.4 * (x2 - 1.5) ** 2 - 0.5
            ))
        )

    def theta2_func(x1, x2):
        return (
                np.log1p(np.exp(
                    np.cos(1.5 * x2) + np.log1p(x1 ** 2) - 0.2
                ))
        )

    theta_functions = [
        theta1_func,
        theta2_func,
    ]

    theta1 = theta_functions[0](x1, x2)
    theta2 = theta_functions[1](x1, x2)

    def model():
        with numpyro.plate(name="data", size=theta1.shape[0]):
            y = numpyro.sample(
                name="y",
                fn=dist.Beta(theta1, theta2),
                rng_key=random.PRNGKey(42)
            )

    model_trace = handlers.trace(model).get_trace()
    y = model_trace["y"]["value"]

    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        sns.histplot(theta1, bins=50, kde=True, ax=axs[0], color='skyblue')
        axs[0].set_title('θ1')
        axs[0].set_xlabel('θ1')
        axs[0].grid(True)
        sns.histplot(theta2, bins=50, kde=True, ax=axs[1], color='salmon')
        axs[1].set_title('θ2')
        axs[1].set_xlabel('θ2')
        axs[1].grid(True)
        sns.histplot(y, bins=50, kde=True, ax=axs[2], color='limegreen')
        axs[2].set_title('Y ~ Beta(θ1, θ2)')
        axs[2].set_xlabel('Y')
        axs[2].grid(True)
        plt.tight_layout()
        plt.show()

        # Visualize marginal feature effects.
        x_grid = np.linspace(lower_bound, upper_bound, 100)
        theta_labels = ["θ1", "θ2"]
        feature_labels = ["x1", "x2"]

        sns.set_style("white", {"axes.grid": True})
        sns.set_palette("Reds")
        fig, axes = plt.subplots(
            n_features,
            len(theta_functions),
            figsize=(6*(len(theta_functions)+1), 6*(n_features+1))
        )
        for i in range(n_features):  # for each feature (x1, x2)
            for j, theta_func in enumerate(theta_functions):  # for each theta (θ1, θ2)
                pdp_vals = []
                for val in x_grid:
                    X_temp = X.copy()
                    X_temp[:, i] = val  # Fix the i-th feature.
                    y_vals = theta_func(X_temp[:, 0], X_temp[:, 1])
                    pdp_vals.append(np.mean(y_vals))

                feature_data = X[:, i]
                theta_data = theta_func(X[:, 0], X[:, 1])

                ax = axes[i, j]
                sns.scatterplot(x=feature_data, y=theta_data, alpha=0.5, ax=ax)
                ax.plot(x_grid, pdp_vals, color="black", linewidth=2)

                if i == 0:
                    ax.set_title(f'{theta_labels[j]}', fontsize=12)
                if j == 0:
                    ax.set_ylabel(f'{feature_labels[i]}', fontsize=12)
                ax.set_xlabel(feature_labels[i])
                ax.grid(True)

        plt.tight_layout()
        plt.show()

    return pd.DataFrame(
        data={
            'x1': x1,
            'x2': x2,
            'theta1': theta1,
            'theta2': theta2,
            'response': y,
        }
    )


def get_poisson_synthetic_data(n_samples: int=3000, seed=42, plot:bool=True):
    lower_bound, upper_bound = 0, 3
    n_features = 2
    X = np.random.uniform(lower_bound, upper_bound, (n_samples, n_features))
    x1, x2 = (
        X[:, 0],
        X[:, 1]
    )

    def theta1_func(x1, x2):
        return (
                np.log1p(np.exp(
                    np.tanh(1.5 * x1 - 2) + 0.5 * np.log1p(x2**2)
                ))
        )

    theta_functions = [
        theta1_func,
    ]

    theta1 = theta_functions[0](x1, x2)

    def model():
        with numpyro.plate(name="data", size=theta1.shape[0]):
            y = numpyro.sample(
                name="y",
                fn=dist.Poisson(theta1),
                rng_key=random.PRNGKey(42)
            )

    model_trace = handlers.trace(model).get_trace()
    y = model_trace["y"]["value"]

    # fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # sns.histplot(theta1, bins=50, kde=True, ax=axs[0], color='skyblue')
    # axs[0].set_title('θ1')
    # axs[0].set_xlabel('θ1')
    # axs[0].grid(True)
    # sns.histplot(y, bins=50, kde=True, ax=axs[2], color='limegreen')
    # axs[1].set_title('Y ~ Beta(θ1, θ2)')
    # axs[1].set_xlabel('Y')
    # axs[1].grid(True)
    # plt.tight_layout()
    # plt.show()

    # Visualize marginal feature effects.
    x_grid = np.linspace(lower_bound, upper_bound, 100)
    theta_labels = ["θ1"]
    feature_labels = ["x1", "x2"]

    sns.set_style("white", {"axes.grid": True})
    sns.set_palette("Reds")
    fig, axes = plt.subplots(
        n_features,
        len(theta_functions),
        figsize=(6*(len(theta_functions)+1), 6*(n_features+1))
    )
    for i in range(n_features):  # for each feature (x1, x2)
        for j, theta_func in enumerate(theta_functions):  # for each theta (θ1, θ2)
            pdp_vals = []
            for val in x_grid:
                X_temp = X.copy()
                X_temp[:, i] = val  # Fix the i-th feature.
                y_vals = theta_func(X_temp[:, 0], X_temp[:, 1])
                pdp_vals.append(np.mean(y_vals))

            feature_data = X[:, i]
            theta_data = theta_func(X[:, 0], X[:, 1])

            ax = axes[i, j] if len(theta_functions) > 1 else axes[i]
            sns.scatterplot(x=feature_data, y=theta_data, alpha=0.5, ax=ax)
            ax.plot(x_grid, pdp_vals, color="black", linewidth=2)

            if i == 0:
                ax.set_title(f'{theta_labels[j]}', fontsize=12)
            if j == 0:
                ax.set_ylabel(f'{feature_labels[i]}', fontsize=12)
            ax.set_xlabel(feature_labels[i])
            ax.grid(True)

    plt.tight_layout()
    plt.show()

    return pd.DataFrame(
        data={
            'x1': x1,
            'x2': x2,
            'theta1': theta1,
            'response': y,
        }
    )


def get_exponential_synthetic_data(n_samples: int=3000, seed=42, plot:bool=True):
    lower_bound, upper_bound = 0, 3
    n_features = 2
    X = np.random.uniform(lower_bound, upper_bound, (n_samples, n_features))
    x1, x2 = (
        X[:, 0],
        X[:, 1]
    )

    def theta1_func(x1, x2):
        return (
                np.log1p(np.exp(
                    np.tanh(1.5 * x1 - 2) + 0.5 * np.log1p(x2**2)
                ))
        )

    theta_functions = [
        theta1_func,
    ]

    theta1 = theta_functions[0](x1, x2)

    def model():
        with numpyro.plate(name="data", size=theta1.shape[0]):
            y = numpyro.sample(
                name="y",
                fn=dist.Exponential(theta1),
                rng_key=random.PRNGKey(42)
            )

    model_trace = handlers.trace(model).get_trace()
    y = model_trace["y"]["value"]

    # fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # sns.histplot(theta1, bins=50, kde=True, ax=axs[0], color='skyblue')
    # axs[0].set_title('θ1')
    # axs[0].set_xlabel('θ1')
    # axs[0].grid(True)
    # sns.histplot(y, bins=50, kde=True, ax=axs[2], color='limegreen')
    # axs[1].set_title('Y ~ Beta(θ1, θ2)')
    # axs[1].set_xlabel('Y')
    # axs[1].grid(True)
    # plt.tight_layout()
    # plt.show()

    # Visualize marginal feature effects.
    x_grid = np.linspace(lower_bound, upper_bound, 100)
    theta_labels = ["θ1"]
    feature_labels = ["x1", "x2"]

    sns.set_style("white", {"axes.grid": True})
    sns.set_palette("Reds")
    fig, axes = plt.subplots(
        n_features,
        len(theta_functions),
        figsize=(6*(len(theta_functions)+1), 6*(n_features+1))
    )
    for i in range(n_features):  # for each feature (x1, x2)
        for j, theta_func in enumerate(theta_functions):  # for each theta (θ1, θ2)
            pdp_vals = []
            for val in x_grid:
                X_temp = X.copy()
                X_temp[:, i] = val  # Fix the i-th feature.
                y_vals = theta_func(X_temp[:, 0], X_temp[:, 1])
                pdp_vals.append(np.mean(y_vals))

            feature_data = X[:, i]
            theta_data = theta_func(X[:, 0], X[:, 1])

            ax = axes[i, j] if len(theta_functions) > 1 else axes[i]
            sns.scatterplot(x=feature_data, y=theta_data, alpha=0.5, ax=ax)
            ax.plot(x_grid, pdp_vals, color="black", linewidth=2)

            if i == 0:
                ax.set_title(f'{theta_labels[j]}', fontsize=12)
            if j == 0:
                ax.set_ylabel(f'{feature_labels[i]}', fontsize=12)
            ax.set_xlabel(feature_labels[i])
            ax.grid(True)

    plt.tight_layout()
    plt.show()

    return pd.DataFrame(
        data={
            'x1': x1,
            'x2': x2,
            'theta1': theta1,
            'response': y,
        }
    )

def get_weibull_synthetic_data(n_samples: int = 3000, seed: int = 42, plot: bool = False):
    """
    Function to generate synthetic data for testing the BayesianNAM model with a Weibull distribution.
    Returns a pandas DataFrame containing the synthetic data.

    Args:
        n_samples (int): Number of samples to generate.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with columns ['x1', 'x2', 'theta1', 'theta2', 'response'].
    """
    # Set seed for reproducibility
    np.random.seed(seed)

    # Generate features
    lower_bound, upper_bound = 0, 3
    n_features = 2
    X = np.random.uniform(lower_bound, upper_bound, (n_samples, n_features))
    x1, x2 = X[:, 0], X[:, 1]

    def theta1_func(x1, x2):
        return (
            np.log1p(np.exp(
                np.sin(1.2 * x1) + 0.4 * (x2 - 1.5) ** 2 - 0.5
            ))
        )

    def theta2_func(x1, x2):
        return (
                np.log1p(np.exp(
                    np.cos(1.5 * x2) + np.log1p(x1 ** 2) - 0.2
                ))
        )

    theta_functions = [
        theta1_func,
        theta2_func,
    ]

    theta1 = theta_functions[0](x1, x2)
    theta2 = theta_functions[1](x1, x2)

    # Sample from Weibull distribution using NumPyro
    def model():
        with numpyro.plate("data", size=n_samples):
            y = numpyro.sample(
                "y",
                dist.Weibull(theta1, theta2),
                rng_key=random.PRNGKey(seed)
            )
        return y

    trace = handlers.trace(model).get_trace()
    y = trace["y"]["value"]

    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        sns.histplot(theta1, bins=50, kde=True, ax=axs[0], color='skyblue')
        axs[0].set_title('θ1')
        axs[0].set_xlabel('θ1')
        axs[0].grid(True)
        sns.histplot(theta2, bins=50, kde=True, ax=axs[1], color='salmon')
        axs[1].set_title('θ2')
        axs[1].set_xlabel('θ2')
        axs[1].grid(True)
        sns.histplot(y, bins=50, kde=True, ax=axs[2], color='limegreen')
        axs[2].set_title('Y ~ Beta(θ1, θ2)')
        axs[2].set_xlabel('Y')
        axs[2].grid(True)
        plt.tight_layout()
        plt.show()

        # Visualize marginal feature effects.
        x_grid = np.linspace(lower_bound, upper_bound, 100)
        theta_labels = ["θ1", "θ2"]
        feature_labels = ["x1", "x2"]

        sns.set_style("white", {"axes.grid": True})
        sns.set_palette("Reds")
        fig, axes = plt.subplots(
            n_features,
            len(theta_functions),
            figsize=(6 * (len(theta_functions) + 1), 6 * (n_features + 1))
        )
        for i in range(n_features):  # for each feature (x1, x2)
            for j, theta_func in enumerate(theta_functions):  # for each theta (θ1, θ2)
                pdp_vals = []
                for val in x_grid:
                    X_temp = X.copy()
                    X_temp[:, i] = val  # Fix the i-th feature.
                    y_vals = theta_func(X_temp[:, 0], X_temp[:, 1])
                    pdp_vals.append(np.mean(y_vals))

                feature_data = X[:, i]
                theta_data = theta_func(X[:, 0], X[:, 1])

                ax = axes[i, j]
                sns.scatterplot(x=feature_data, y=theta_data, alpha=0.5, ax=ax)
                ax.plot(x_grid, pdp_vals, color="black", linewidth=2)

                if i == 0:
                    ax.set_title(f'{theta_labels[j]}', fontsize=12)
                if j == 0:
                    ax.set_ylabel(f'{feature_labels[i]}', fontsize=12)
                ax.set_xlabel(feature_labels[i])
                ax.grid(True)

        plt.tight_layout()
        plt.show()

    # Assemble into DataFrame
    return pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'theta1': theta1,
        'theta2': theta2,
        'response': y,
    })


def plot_feature_contributions(
        num_features: dict,
        cat_features: dict,
        interaction_features: dict,
        submodel_contributions: dict,
        num_outputs: int
):
    sns.set_style("white")
    sns.set_palette("Greens")

    for feature_dict in [num_features, cat_features, interaction_features]:
        if not feature_dict:
            continue

        num_plots = len(feature_dict)
        fig, ax = plt.subplots(
            nrows=num_plots, ncols=num_outputs,
            figsize=(6 * num_outputs, 6 * num_plots),
            squeeze=False
        )
        for i, (feature_name, feature_array) in enumerate(feature_dict.items()):
            feature_values = np.array(feature_array).flatten()  # Convert JAX array to NumPy

            # Shape: [num_mcmc_samples, batch_size, network_output_dim]
            contributions = submodel_contributions[feature_name]

            # [batch_size, network_output_dim]
            mean_contribution_all_params = contributions.mean(axis=0)
            for j, target_name in zip(
                    range(num_outputs), ["Mean", "Std. Deviation"]
            ):
                mean_param_contribution = mean_contribution_all_params[:, j] \
                    if num_outputs > 1 else mean_contribution_all_params

                sorted_idx = np.argsort(feature_values)
                feature_values_sorted = feature_values[sorted_idx]
                mean_param_contribution_sorted = mean_param_contribution[sorted_idx]

                ax_to_plot = ax[i, j]
                uncertainty = np.std(
                    submodel_contributions[feature_name][:, :, j],
                    axis=0
                )[sorted_idx] \
                    if num_outputs > 1 \
                    else np.std(
                    submodel_contributions[feature_name],
                    axis=0
                )[sorted_idx]
                for ci_multiplier, alpha in zip([3, 2, 1], [0.3, 0.6, 0.9]):
                    ax_to_plot.fill_between(
                        feature_values_sorted,
                        mean_param_contribution_sorted - ci_multiplier * uncertainty,
                        mean_param_contribution_sorted + ci_multiplier * uncertainty,
                        label=f"Epistemic Uncertainty - {ci_multiplier} Std. Deviations"
                    )
                    ax_to_plot.plot(
                        feature_values_sorted,
                        mean_param_contribution_sorted - ci_multiplier * uncertainty,
                        color='black',
                        linestyle='dashed',
                        alpha=0.2
                    )
                    ax_to_plot.plot(
                        feature_values_sorted,
                        mean_param_contribution_sorted + ci_multiplier * uncertainty,
                        color='black',
                        linestyle='dashed',
                        alpha=0.2
                    )

                # num_bins = 30
                # counts, bin_edges = np.histogram(feature_values, bins=num_bins)
                # norm_counts = counts / counts.max()
                # fixed_height = ax_to_plot.get_ylim()[1] - ax_to_plot.get_ylim()[0]
                # for k in range(num_bins):
                #     ax_to_plot.bar(
                #         bin_edges[k],
                #         height=fixed_height,
                #         bottom=ax_to_plot.get_ylim()[0],
                #         width=bin_edges[k + 1] - bin_edges[k],
                #         color=plt.cm.Blues(norm_counts[k]),
                #         alpha=0.25
                #     )

                sns.lineplot(
                    x=feature_values_sorted,
                    y=mean_param_contribution_sorted,
                    label="Mean",
                    ax=ax_to_plot,
                    color="black",
                    drawstyle="steps" if np.all(
                        np.isin(feature_values, [0, 1])
                    ) else None
                )

                ax_to_plot.set_xlabel(f"{feature_name}", fontsize=12)
                ax_to_plot.set_ylabel(f"{target_name}", fontsize=12)
                ax_to_plot.set_title(f"Out-of-sample {target_name} Fit", fontsize=12)
                ax_to_plot.legend(
                    # loc='upper left',
                    fontsize=12, frameon=False
                )
                ax_to_plot.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    warmstart_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "namgcv",
        "bnam_de_warmstart_checkpoints",
    )
    # find the "tree" file and all ".npz" files in the directory and delete them
    for root, dirs, files in os.walk(warmstart_path):
        for file in files:
            if file.endswith(".npz") or file.endswith("tree"):
                os.remove(os.path.join(root, file))

    from namgcv.configs.experimental.small_synthetic_nam_beta import DefaultBayesianNAMConfig
    from namgcv.configs.experimental.small_synthetic_nn_beta import DefaultBayesianNNConfig

    # data = get_invgamma_synthetic_data(n_samples=3000, plot=False)
    data = get_beta_synthetic_data(n_samples=1000, plot=False)
    # data = get_gaussian_synthetic_data(n_samples=3000)
    # data = get_weibull_synthetic_data(n_samples=3000, plot=True)
    # data = get_poisson_synthetic_data(n_samples=3000, plot=True)
    # data = get_exponential_synthetic_data(n_samples=3000, plot=True)

    X = data.drop(columns=['response'] + [col for col in data.columns if "theta" in col])
    y = data['response']

    # One hot encode the categorical features.
    try:
        cat_features = pd.get_dummies(
            data=X[[col for col in X.columns if X[col].dtype == "category"]]
        ).astype(int)
    except ValueError:  # No categorical features.
        cat_features = pd.DataFrame()

    num_features = X[[col for col in X.columns if X[col].dtype != "category"]]
    print(f"Categorical features: {cat_features.columns.tolist()}")
    print(f"Numeric features: {num_features.columns.tolist()}")

    # --- Model Setup ---
    num_outputs = 1
    cat_feature_info, cat_feature_inputs = {}, {}
    for cat_col in [col for col in X.columns if X[col].dtype == "category"]:
        all_cols_for_feature = [
            col for col in cat_features.columns if col.startswith(cat_col)
        ]
        cat_feature_info[cat_col] = {
            "input_dim": len(all_cols_for_feature),
            "output_dim": num_outputs
        }
        cat_feature_inputs[cat_col] = jnp.array(
            cat_features[all_cols_for_feature]
        )

    num_feature_info = {
        feature_name: {
            "input_dim": 1,
            "output_dim": num_outputs
        } for feature_name in num_features.columns
    }
    num_feature_inputs = {
        feature_name: jnp.array(
            num_features.loc[:, feature_name]
        ) for feature_name in num_feature_info.keys()
    }

    numpyro.set_host_device_count(
        DefaultBayesianNAMConfig().num_chains
    )
    model = BayesianNAM(
        cat_feature_info=cat_feature_info,
        num_feature_info=num_feature_info,
        config=DefaultBayesianNAMConfig(),
        subnetwork_config=DefaultBayesianNNConfig(),
        link_1=lambda x: jax.nn.softplus(jnp.clip(x, -8,8)),
        link_2=lambda x: jax.nn.softplus(jnp.clip(x, -8,8)),
    )

    # --- Training ---
    CV = False
    if CV:
        cv_results = model.cross_validation(
            num_features=num_feature_inputs,
            cat_features=cat_feature_inputs,
            target=jnp.array(y),
        )
    else:
        data_loader = model.train_model(
            num_features={
                feature_name: jnp.array(
                    num_features.iloc[:, col_idx]
                ) for col_idx, feature_name in enumerate(
                    num_features.columns
                )
            },
            cat_features={
                feature_name: jnp.array(
                    cat_features.iloc[:, col_idx]
                ) for col_idx, feature_name in enumerate(
                    cat_features.columns
                )
            },
            target=jnp.array(y)
        )

    pass

    for bayesian_flag in [False, True]:
        final_params, submodel_contributions = model.predict(
            data_loader=data_loader,
            bayesian_sampling=bayesian_flag
        )

        batch_iter = data_loader.iter(
            "test",
            batch_size=None
        )
        data_dict = next(batch_iter)  # First and only batch.
        num_features = data_dict["feature"]["numerical"]
        cat_features = data_dict["feature"]["categorical"]
        target = data_dict["target"]

        interaction_feature_information = {}
        all_features = {**num_features, **cat_features}
        for interaction_name in submodel_contributions.keys():
            if ":" not in interaction_name:
                continue

            feature_names = interaction_name.split(":")
            interaction_feature_information[interaction_name] = jnp.concatenate(
                [jnp.expand_dims(all_features[name], axis=-1) for name in feature_names],
                axis=-1
            )

        plot_feature_contributions(
            num_features=num_features,
            cat_features=cat_features,
            interaction_features=interaction_feature_information,
            submodel_contributions=submodel_contributions,
            num_outputs=num_outputs
        )

        import numpyro.distributions as dist
        log_probs = dist.Beta(
            final_params[..., 0].mean(axis=0),
            final_params[..., 1].mean(axis=0)
        ).log_prob(target)  # shape: (num_samples, num_data)
        nll = -jnp.mean(log_probs, axis=-1)
        print("Negative Log-Likelihood:", nll)
        # from properscoring import crps_gaussian
        # crps = crps_gaussian(
        #     x=target,
        #     mu=final_params[..., 0].mean(axis=0),
        #     sig=final_params[..., 1].mean(axis=0)
        # )
        # print("CRPS:", crps)

        # from properscoring import crps_quadrature
        # import scipy
        # crps = np.mean([
        #     crps_quadrature(
        #         x=target[i],
        #         cdf_or_dist= scipy.stats.invgamma(
        #             a=final_params[..., 0].mean(axis=0)[i],
        #             scale=final_params[..., 1].mean(axis=0)[i]
        #         )
        #     ) for i in range(len(target))
        # ])
        # print("CRPS:", crps)

        import xarray
        from scores.probability import crps_cdf

        fcst_thresholds = np.linspace(0, 30, len(target))
        crps_scores_orig = [crps_cdf(
            fcst=xarray.DataArray(
                coords={'temperature': fcst_thresholds},
                data=scipy.stats.invgamma(
                    a=final_params[..., 0].mean(axis=0)[i],
                    scale=final_params[..., 1].mean(axis=0)[i]
                ).cdf(fcst_thresholds)
            ),
            obs=xarray.DataArray(target[i]),
            threshold_dim="temperature"
        ) for i in range(len(target))]

        from arviz.sel_utils import xarray_to_ndarray
        crps_scores = pd.Series(
            data=[
                xarray_to_ndarray(crps_scores_orig[i])[1].flatten()[0]
                for i in range(len(crps_scores_orig))
            ], name="crps"
        )
        crps_score = np.mean(crps_scores[crps_scores < 20])
        print("CRPS:", crps_score)


    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from itertools import combinations

    from sympy.printing.pretty.pretty_symbology import line_width

    posterior_param_samples_dict = model._get_posterior_param_samples(
        group_by_chain=True
    )
    # number of random weights to pick per layer
    K = 2
    # fixed seed for reproducibility
    SEED = 422
    rng = np.random.default_rng(SEED)

    sns.set_style("white")
    for layer_num in [0, 1, 2, 3, 4]:
        for posterior_params_name, posterior_param_samples in posterior_param_samples_dict.items():
            if not posterior_param_samples or "weight" not in posterior_params_name:
                continue

            feature_names = {k.split("/")[0] for k in posterior_param_samples}
            for feature_name in feature_names:
                param_names = [k for k in posterior_param_samples if feature_name in k]

                for param_name in param_names:
                    if f"dense_{layer_num}" not in param_name or "scale" in param_name:
                        continue

                    raw = posterior_param_samples[param_name]
                    # ensure shape (chains, draws, n_wts, n_outs)
                    if raw.ndim == 3:
                        raw = raw[None, ...]
                    n_chains, n_draws, n_wts, n_outs = raw.shape

                    total = n_wts * n_outs
                    assert isinstance(K, int), "K must be int"
                    assert 2 <= K <= total, f"K={K} out of range [2, {total}]"

                    # pick the same K weights every run
                    idx = rng.choice(total, size=K, replace=False)
                    names = [
                        f"Dense Layer {layer_num} - Neuron {i} (Out {j})"
                        for j in range(n_outs)
                        for i in range(n_wts)
                    ]
                    picked_names = [names[i] for i in idx]

                    flat = raw.reshape(n_chains, n_draws, total)
                    sampled = flat[:, :, idx]

                    palette = sns.color_palette("Greens", n_chains)

                    # loop over all pairs among the K
                    for a, b in combinations(range(K), 2):
                        g = sns.JointGrid(height=6)

                        if n_chains == 1:
                            # single‐chain: fill + black border + marginals
                            x = sampled[0, :, a]
                            y = sampled[0, :, b]

                            g.x, g.y = x, y
                            g.plot_joint(
                                sns.kdeplot,
                                fill=True, cmap="Greens", alpha=0.8
                            )
                            g.plot_joint(
                                sns.kdeplot,
                                fill=False, color="green", levels=10, linewidths=1.0
                            )
                            g.plot_marginals(
                                sns.kdeplot,
                                fill=True, color=palette[0]
                            )

                        else:
                            # multi‐chain: overlay each chain
                            for c in range(n_chains):
                                x = sampled[c, :, a]
                                y = sampled[c, :, b]

                                g.x, g.y = x, y
                                g.plot_joint(
                                    sns.kdeplot,
                                    fill=False, color=palette[c], levels=10, linewidths=1.0
                                )
                                g.plot_marginals(
                                    sns.kdeplot,
                                    fill=True, color=palette[c]
                                )

                        # labels & title
                        g.ax_joint.set_xlabel(picked_names[a])
                        g.ax_joint.set_ylabel(picked_names[b])
                        suffix = "" if n_chains == 1 else " (all chains)"
                        plt.suptitle(
                            f"{feature_name} BNN Bivariate Marginal Posterior{suffix}"
                        )
                        g.ax_joint.grid(True)
                        plt.tight_layout()
                        plt.show()

        print(f"Layer {layer_num} done.")
    print("Marginal posteriors done.")


    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from itertools import combinations

    # ───── User settings ─────
    layer_nums = [2, 3, 4]
    output_indices = [0]
    reduce_along_input = "mean"  # or "sum", "max", …
    K = 2  # number of hidden units to plot per layer
    # SEED                = 42
    # ─────────────────────────

    # assume group_by_chain is defined somewhere (e.g. True or False)
    posterior_param_samples_dict = model._get_posterior_param_samples(
        group_by_chain=False
    )

    rng = np.random.default_rng(SEED)
    sns.set_style("white")


    def normalize(arr):
        """
        Turn a 3D array (S, H, D) into (1, S, H, D),
        or pass through a 4D array (C, S, H, D).
        Returns (new_arr, n_chains, n_draws, [H, D]).
        """
        if arr.ndim == 4:
            nc, nd, H, D = arr.shape
            return arr, nc, nd, [H, D]
        elif arr.ndim == 3:
            nd, H, D = arr.shape
            return arr[None, ...], 1, nd, [H, D]
        else:
            raise ValueError(f"Expected 3D or 4D, got {arr.ndim}D")


    for layer_num in layer_nums:
        prev_layer = layer_num - 1

        for output_index in output_indices:
            for _, posterior_param_samples in posterior_param_samples_dict.items():
                # only look at dicts containing weights
                if not posterior_param_samples or "weight" not in _:
                    continue

                feature_names = {k.split("/")[0] for k in posterior_param_samples}
                for feature in feature_names:
                    prev_key = f"{feature}/dense_{prev_layer}_kernel"
                    last_key = f"{feature}/dense_{layer_num}_kernel"
                    if prev_key not in posterior_param_samples or last_key not in posterior_param_samples:
                        continue

                    W_prev_raw = posterior_param_samples[prev_key]
                    W_last_raw = posterior_param_samples[last_key]

                    # normalize shapes
                    W_prev, nc1, nd1, [n_hidden, n_in] = normalize(W_prev_raw)
                    W_last, nc2, nd2, [_, n_out] = normalize(W_last_raw)
                    assert nc1 == nc2, "Chain‐count mismatch"
                    assert nd1 == nd2, "Draw‐count mismatch"
                    nc, nd = nc1, nd1

                    # pick one output head & form Hadamard u
                    w_out = W_last[..., output_index]  # (nc, nd, n_hidden)
                    u = w_out[..., None] * W_prev  # (nc, nd, n_hidden, n_in)

                    # reduce along input dim
                    if reduce_along_input == "mean":
                        u = u.mean(axis=-1)
                    elif reduce_along_input == "sum":
                        u = u.sum(axis=-1)
                    elif reduce_along_input == "max":
                        u = u.max(axis=-1)
                    else:
                        raise ValueError(f"Unknown reduction {reduce_along_input!r}")
                    # now u.shape == (nc, nd, n_hidden)

                    # ─── sample K hidden units ───
                    assert isinstance(K, int), "K must be an integer"
                    assert 2 <= K <= n_hidden, f"K={K} must be in [2, {n_hidden}]"
                    hidden_idx = rng.choice(n_hidden, size=K, replace=False)
                    u = u[:, :, hidden_idx]  # (nc, nd, K)
                    col_names = [f"u[{h}]" for h in hidden_idx]

                    # palette for chains
                    palette = sns.color_palette("Reds", nc)

                    # plot every pair among our K units
                    for a, b in combinations(range(K), 2):
                        g = sns.JointGrid(height=6)

                        if nc == 1:
                            # single chain: fill + black border + red marginals
                            x = u[0, :, a]
                            y = u[0, :, b]
                            g.x, g.y = x, y

                            g.plot_joint(
                                sns.kdeplot,
                                fill=True, cmap="Reds", alpha=0.8
                            )
                            g.plot_joint(
                                sns.kdeplot,
                                fill=False, color="red", levels=10, linewidths=1.0
                            )
                            g.plot_marginals(
                                sns.kdeplot,
                                fill=True, color="red"
                            )

                        else:
                            # multiple chains: overlay each chain's contour + marginals
                            for c in range(nc):
                                x = u[c, :, a]
                                y = u[c, :, b]
                                g.x, g.y = x, y

                                g.plot_joint(
                                    sns.kdeplot,
                                    fill=False, color=palette[c], levels=10, linewidths=1.0
                                )
                                g.plot_marginals(
                                    sns.kdeplot,
                                    fill=True, color=palette[c]
                                )

                        # labels, title, grid
                        g.ax_joint.set_xlabel(col_names[a])
                        g.ax_joint.set_ylabel(col_names[b])

                        title = (
                            f"{feature} – Hadamard (dense {prev_layer} ⊙ dense {layer_num})"
                        )
                        if nc > 1:
                            title += "  (all chains)"
                        plt.suptitle(title, fontsize=12)

                        g.ax_joint.grid(True)
                        plt.tight_layout()
                        plt.show()

        print(f"Layer {layer_num} done.")
    print("Hadamard Marginal posteriors done.")


