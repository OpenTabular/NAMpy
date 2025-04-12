import os

from IPython.core.pylabtools import figsize

num_chains = 10
n_devices = min(os.cpu_count(), num_chains)
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={n_devices}'

import pandas as pd
import numpy as np
import jax.numpy as jnp

import numpyro

import matplotlib.pyplot as plt
import seaborn as sns

import jax
from jax.lib import xla_bridge

from scipy.stats import norm

print(xla_bridge.get_backend().platform)
print(f"Default backend for JAX: {jax.default_backend()}")
print(
    f"Number of devices available on default backend: "
    f"{jax.local_device_count(backend=jax.default_backend())}"
)

from namgcv.basemodels.bnam import BayesianNAM
from namgcv.configs.experimental.small_synthetic_nam import DefaultBayesianNAMConfig
from namgcv.configs.experimental.small_synthetic_nn import DefaultBayesianNNConfig

# --- DATA ---
def get_independent_synthetic_data(n_samples: int=3000,  seed=42):
    """
    Function to generate synthetic data for testing the BayesianNAM model.
    Returns a pandas DataFrame containing the synthetic data.

    Args:
        n_samples (int): Number of samples to generate.

    Returns:
        pd.DataFrame: DataFrame containing the synthetic data.
    """

    np.random.seed(seed)

    # Generate 5 features uniformly distributed in [0, 1]
    X = np.random.uniform(0, 1, (n_samples, 5))
    x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

    # Small constant to avoid division by zero issues
    eps = 1e-6

    # Compute the synthetic distributional parameters:
    # θ(1) = (30/13) * x1 * (((3*x2 + 1.5)^(-2) * sin(x3^2))^(-1)) + (113/115)*x4 + 0.1*x5
    # Note: (((3*x2 + 1.5)^(-2) * sin(x3^2))^(-1)) is equivalent to (3*x2 + 1.5)**2 / sin(x3**2)
    theta1 = (
            (30 / 13) * x1 * (((3 * x2 + 1.5) ** (-2) * (np.sin(x3 ** 2) + eps)) ** (-1))
             + (113 / 115) * x4 + 0.1 * x5
    )

    # θ(2) = exp(-0.0035*x1 + (x2 - 0.23)**2 - 1.42*x3) + 0.0001*x4
    theta2 = np.exp(-0.0035 * x1 + (x2 - 0.23) ** 2 - 1.42 * x3) + 0.0001 * x4

    # θ(3) = (1/42) * (4*x1 - 90*x2)
    theta3 = (1 / 42) * (4 * x1 - 90 * x2)

    # θ(4) = exp(0.0323*x2 + 0.0123 - 0.0234*x4)
    theta4 = np.exp(0.0323 * x2 + 0.0123 - 0.0234 * x4)

    # For demonstration purposes, we generate the response variable y from a normal distribution
    # with mean theta1 and standard deviation theta2.
    y = norm.rvs(loc=theta1, scale=theta2, size=n_samples)

    # Package everything into a DataFrame
    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,
        'x5': x5,
        # 'theta1': theta1,
        # 'theta2': theta2,
        # 'theta3': theta3,
        # 'theta4': theta4,
        'y': y
    })

    return df


def plot_synthetic_data(
        true_effects: dict,
        palette: str = "Reds"
):
    """
    Function to plot the synthetic data generated for testing the BayesianNAM model.

    Parameters
    ----------
    true_effects: dict containing the true effects of the features.
    response: jnp.ndarray containing the response variable.
    """

    sns.set_style("white")
    sns.set_palette(palette)

    single_true_effects = {
        feature_name: feature_data
        for feature_name, feature_data in true_effects.items()
        if ":" not in feature_name
    }
    n_cols = 2
    fig, axes = plt.subplots(
        nrows=len(single_true_effects) ,
        ncols=n_cols,
        figsize=(6 * n_cols, 6 * len(single_true_effects))
    )
    for i, feature_name in enumerate(single_true_effects.keys()):
        x_vals = single_true_effects[feature_name]["feature"]
        y_mean = single_true_effects[feature_name]["response"]

        sort_idx = np.argsort(x_vals)
        x_sorted = x_vals[sort_idx]
        y_sorted = y_mean[sort_idx]

        ax_to_plot = axes[i, 0] if len(single_true_effects) > 1 else axes[0]
        for ci_multiplier, alpha in zip([3, 2, 1], [0.2, 0.4, 0.8]):
            ax_to_plot.fill_between(
                x_sorted,
                y_sorted
                - ci_multiplier
                * true_effects[feature_name]["noise_parameters"]["scale"][sort_idx],
                y_sorted
                + ci_multiplier
                * true_effects[feature_name]["noise_parameters"]["scale"][sort_idx],
                alpha=alpha
            )
            # Plot outlines for the filled areas.
            ax_to_plot.plot(
                x_sorted,
                y_sorted - ci_multiplier
                * true_effects[feature_name]["noise_parameters"]["scale"][sort_idx],
                color='black',
                linestyle='dashed',
                alpha=0.2
            )
            ax_to_plot.plot(
                x_sorted,
                y_sorted
                + ci_multiplier
                * true_effects[feature_name]["noise_parameters"]["scale"][sort_idx],
                color='black',
                linestyle='dashed',
                alpha=0.2
            )

        sns.scatterplot(
            x=single_true_effects[feature_name]["feature"][sort_idx],
            y=single_true_effects[feature_name]["response"][sort_idx]
              + single_true_effects[feature_name]["noise"][sort_idx],
            label="Noisy Response",
            ax=ax_to_plot
        )
        sns.lineplot(
            x=single_true_effects[feature_name]["feature"][sort_idx],
            y=single_true_effects[feature_name]["response"][sort_idx],
            color="black",
            label="True Mean Marginal Effect",
            ax=ax_to_plot,
        )
        ax_to_plot.set_title(f"{feature_name} Effect", fontsize=14)
        ax_to_plot.set_xlabel(f"{feature_name}", fontsize=12)
        ax_to_plot.set_ylabel("y", fontsize=12)
        ax_to_plot.legend(loc='upper center')
        ax_to_plot.grid(True)

        # Plot noise intervals.
        ax_to_plot = axes[i, 1] if len(single_true_effects) > 1 else axes[1]
        ax_to_plot.axhline(0, color='black', label="True Noise Mean")
        for ci_multiplier, alpha in zip([3, 2, 1], [0.2, 0.4, 0.8]):
            ax_to_plot.fill_between(
                x_sorted,
                -ci_multiplier * true_effects[feature_name]["noise_parameters"]["scale"][sort_idx],
                ci_multiplier * true_effects[feature_name]["noise_parameters"]["scale"][sort_idx],
                alpha=alpha
            )
            # Plot outlines for the filled areas.
            ax_to_plot.plot(
                x_sorted,
                -ci_multiplier * true_effects[feature_name]["noise_parameters"]["scale"][sort_idx],
                color='black',
                linestyle='dashed',
                alpha=0.2
            )
            ax_to_plot.plot(
                x_sorted,
                ci_multiplier * true_effects[feature_name]["noise_parameters"]["scale"][sort_idx],
                color='black',
                linestyle='dashed',
                alpha=0.2
            )

        sns.scatterplot(
            x=single_true_effects[feature_name]["feature"][sort_idx],
            y=single_true_effects[feature_name]["noise"][sort_idx],
            label="Sampled Noise",
            ax=ax_to_plot
        )

        ax_to_plot.set_title(f"{feature_name} Noise", fontsize=14)
        ax_to_plot.set_xlabel(f"{feature_name}", fontsize=12)
        ax_to_plot.set_ylabel(r"$\epsilon$", fontsize=12)
        ax_to_plot.legend(loc='upper center')
        ax_to_plot.grid(True)

    plt.legend(loc='upper center')
    plt.tight_layout()
    plt.show()

    fig, ax_to_plot  = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    x = np.random.uniform(low=-1.5, high=1.5, size=len(x_sorted))
    sorted_idx = np.argsort(x)
    y = (x ** 2 + np.sin(4 * x))
    sns.lineplot(
        x=x[sorted_idx],
        y=y[sorted_idx],
        label="Target",
        ax=ax_to_plot,
        color="black"
    )
    var_y = 0.01 + np.exp(x) / 10 + 0.1 + np.square(x)
    for ci_multiplier, alpha in zip([3, 2, 1], [0.2, 0.4, 0.8]):
        ax_to_plot.plot(
            x[sorted_idx],
            y[sorted_idx] - ci_multiplier * var_y[sorted_idx],
            color='black',
            linestyle='dashed',
            alpha=0.2
        )
        ax_to_plot.plot(
            x[sorted_idx],
            y[sorted_idx] + ci_multiplier * var_y[sorted_idx],
            color='black',
            linestyle='dashed',
            alpha=0.2
        )
        # Shade the area between the noise intervals.
        ax_to_plot.fill_between(
            x[sorted_idx],
            y[sorted_idx] - ci_multiplier * var_y[sorted_idx],
            y[sorted_idx] + ci_multiplier * var_y[sorted_idx],
            alpha=alpha
        )
    ax_to_plot.set_title(f"Target", fontsize=14)
    ax_to_plot.set_xlabel(f"x", fontsize=12)
    ax_to_plot.set_ylabel("y", fontsize=12)
    ax_to_plot.legend(loc='upper center')
    ax_to_plot.grid(True)

    plt.legend(loc='upper center')
    plt.tight_layout()
    plt.show()


def plot_predictions(
        num_features: dict,
        cat_features: dict,
        interaction_features: dict,
        submodel_contributions: dict,
        num_outputs: int,
        target
):
    sns.set_style("white")
    sns.set_palette("Reds")

    for feature_dict in [num_features, cat_features, interaction_features]:
        if not feature_dict:
            continue

        num_plots = len(feature_dict)
        fig, ax = plt.subplots(
            nrows=num_plots, ncols=1,
            figsize=(6, 6 * num_plots),
            squeeze=False
        )
        for i, (feature_name, feature_array) in enumerate(feature_dict.items()):
            ax_to_plot = ax[i, 0]

            feature_values = np.array(feature_array).flatten()
            # [num_mcmc_samples, batch_size, network_output_dim]
            contributions = submodel_contributions[feature_name]
            # [batch_size, network_output_dim]
            mean_contribution_all_params = contributions.mean(axis=0)

            mean_param_contribution = mean_contribution_all_params[:, 0] \
                if num_outputs > 1 else mean_contribution_all_params

            sorted_idx = np.argsort(feature_values)
            feature_values_sorted = feature_values[sorted_idx]
            mean_param_contribution_sorted = mean_param_contribution[sorted_idx]

            sns.lineplot(
                x=feature_values_sorted,
                y=mean_param_contribution_sorted,
                label="Mean Output Parameter Contribution",
                ax=ax_to_plot,
                color="black"
            )

            if num_outputs > 1:
                uncertainty = mean_contribution_all_params[:, 1]
                uncertainty_sorted = uncertainty[sorted_idx]
                for ci_multiplier, alpha in zip([3, 2, 1], [0.3, 0.6, 0.9]):
                    ax_to_plot.fill_between(
                        feature_values_sorted,
                        mean_param_contribution_sorted - ci_multiplier * uncertainty_sorted,
                        mean_param_contribution_sorted + ci_multiplier * uncertainty_sorted,
                        alpha=alpha,
                        label=f"Aleatoric Uncertainty - {ci_multiplier} Std. Deviations"
                    )
                    ax_to_plot.plot(
                        feature_values_sorted,
                        mean_param_contribution_sorted - ci_multiplier * uncertainty_sorted,
                        color='black',
                        linestyle='dashed',
                        alpha=0.2
                    )
                    ax_to_plot.plot(
                        feature_values_sorted,
                        mean_param_contribution_sorted + ci_multiplier * uncertainty_sorted,
                        color='black',
                        linestyle='dashed',
                        alpha=0.2
                    )
            else:
                pass  # no aleatoric uncertainty to plot in mean regression.

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
            #         alpha=0.2
            #     )

            # sns.scatterplot(
            #     x=feature_values_sorted,
            #     y=target[sorted_idx],
            #     label="Target"
            # )

            ax_to_plot.set_xlabel(f"{feature_name}", fontsize=12)
            ax_to_plot.set_ylabel("Feature Contribution", fontsize=12)
            ax_to_plot.set_title(f"Feature Contribution for {feature_name}", fontsize=12)
            ax_to_plot.legend(loc='upper center', fontsize=12, frameon=False)
        
        fig.show()


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
            for j in range(num_outputs):
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
                ax_to_plot.fill_between(
                    feature_values_sorted,
                    mean_param_contribution_sorted - 1.96 * uncertainty,
                    mean_param_contribution_sorted + 1.96 * uncertainty,
                    alpha=0.8,
                    label="Epistemic Uncertainty - 95% Interval"
                )
                num_bins = 30
                counts, bin_edges = np.histogram(feature_values, bins=num_bins)
                norm_counts = counts / counts.max()
                fixed_height = ax_to_plot.get_ylim()[1] - ax_to_plot.get_ylim()[0]
                for k in range(num_bins):
                    ax_to_plot.bar(
                        bin_edges[k],
                        height=fixed_height,
                        bottom=ax_to_plot.get_ylim()[0],
                        width=bin_edges[k + 1] - bin_edges[k],
                        color=plt.cm.Blues(norm_counts[k]),
                        alpha=0.3
                    )

                sns.lineplot(
                    x=feature_values_sorted,
                    y=mean_param_contribution_sorted,
                    label="Mean Output Parameter Contribution",
                    ax=ax_to_plot,
                    color="black",
                    drawstyle="steps" if np.all(
                        np.isin(feature_values, [0, 1])
                    ) else None
                )

                ax_to_plot.set_xlabel(f"{feature_name}", fontsize=12)
                ax_to_plot.set_ylabel("Feature Contribution", fontsize=12)
                ax_to_plot.set_title(f"Feature Contribution for {feature_name}", fontsize=12)
                ax_to_plot.legend(loc='upper center', fontsize=12, frameon=False)
                ax_to_plot.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    data = get_independent_synthetic_data(n_samples=3000)
    # plot_synthetic_data(true_effects=true_effects)

    X = data.drop(columns=['y'])
    y = data['y']

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
    num_outputs = 2
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
        subnetwork_config=DefaultBayesianNNConfig()
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
            target=jnp.array(y),
        )

    pass

    final_params, submodel_contributions = model.predict(data_loader=data_loader)

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

    # plot_feature_contributions(
    #     num_features=num_features,
    #     cat_features=cat_features,
    #     interaction_features=interaction_feature_information,
    #     submodel_contributions=submodel_contributions,
    #     num_outputs=num_outputs
    # )

    # plot_predictions(
    #     num_features=num_features,
    #     cat_features=cat_features,
    #     interaction_features=interaction_feature_information,
    #     submodel_contributions=submodel_contributions,
    #     num_outputs=num_outputs,
    #     target=target
    # )

    import properscoring
    score, score_grad = properscoring.crps_gaussian(
        x=target,
        mu=final_params[..., 0].mean(axis=0),
        sig=final_params[..., 1].mean(axis=0),
        grad=True
    )
    print(f"CRPS: {score.mean(axis=0)} | CRPS Gradient (mu, sigma): {score_grad.mean(axis=1)}")

    import numpyro.distributions as dist
    log_probs = dist.Normal(
        final_params[..., 0].mean(axis=0),
        final_params[..., 1].mean(axis=0)
    ).log_prob(target)  # shape: (num_samples, num_data)
    nll = -jnp.mean(
        jnp.sum(log_probs, axis=-1)
    )
    print("Negative Log-Likelihood:", nll)

    pass

    # import cloudpickle
    # import lzma
    #
    # model_name = "small_synthetic_nam_100warmup_1000sample_8chains_16x16_relu_nonhierarchical_isotropic_0mean_5scale"
    # with lzma.open(
    #     filename=f"{model_name}.pkl.xz",
    #     mode="wb"
    # ) as f:
    #     cloudpickle.dump(model, f)
    #
    # with lzma.open(
    #     filename=f"{model_name}.pkl.xz",
    #     mode="rb"
    # ) as f:
    #     loaded_model = cloudpickle.load(f)