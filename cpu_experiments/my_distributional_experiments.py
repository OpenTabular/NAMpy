import os

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

print(xla_bridge.get_backend().platform)
print(f"Default backend for JAX: {jax.default_backend()}")
print(
    f"Number of devices available on default backend: "
    f"{jax.local_device_count(backend=jax.default_backend())}"
)

from namgcv.basemodels.bnam import BayesianNAM
from namgcv.configs.experimental.small_synthetic_nam import DefaultBayesianNAMConfig
from namgcv.configs.experimental.small_synthetic_nn import DefaultBayesianNNConfig

from scipy import stats


def get_independent_synthetic_data(n_samples: int=3000,  seed=42):
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

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(theta1, bins=50, kde=True, ax=axs[0], color='skyblue')
    axs[0].set_title('θ₁')
    axs[0].set_xlabel('θ₁')
    axs[0].grid(True)
    sns.histplot(theta2, bins=50, kde=True, ax=axs[1], color='salmon')
    axs[1].set_title('θ₂')
    axs[1].set_xlabel('θ₂')
    axs[1].grid(True)
    sns.histplot(y, bins=50, kde=True, ax=axs[2], color='limegreen')
    axs[2].set_title('Y ~ InvGamma(θ₁, θ₂)')
    axs[2].set_xlabel('Y')
    axs[2].grid(True)
    plt.tight_layout()
    plt.show()

    # Visualize marginal feature effects.
    x_grid = np.linspace(lower_bound, upper_bound, 100)
    theta_labels = ["θ₁", "θ₂"]
    feature_labels = ["x₁", "x₂"]

    sns.set_style("white", {"axes.grid": True})
    sns.set_palette("Reds")
    fig, axes = plt.subplots(
        n_features,
        len(theta_functions),
        figsize=(6*len(theta_functions), 6*n_features)
    )
    for i in range(n_features):  # for each feature (x1, x2)
        for j, theta_func in enumerate(theta_functions):  # for each theta (θ₁, θ₂)
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
    data = get_independent_synthetic_data(n_samples=1000)

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
        link_1 = lambda x: jax.nn.softplus(
            jnp.clip(x, -10, 10)
        ),
        link_2 = lambda x: jax.nn.softplus(
            jnp.clip(x, -10, 10)
        )
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

    plot_feature_contributions(
        num_features=num_features,
        cat_features=cat_features,
        interaction_features=interaction_feature_information,
        submodel_contributions=submodel_contributions,
        num_outputs=num_outputs
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    theta_1 = final_params.mean(axis=0)[..., 0]
    theta_2 = final_params.mean(axis=0)[..., 1]
    x = num_features["x1"]
    sorted_idx = np.argsort(x)
    # plot theta1 and theta2 as functions of x
    ax[0].plot(
        x[sorted_idx],
        theta_1[sorted_idx],
        label="Theta 1",
    )
    ax[1].plot(
        x[sorted_idx],
        theta_2[sorted_idx],
        label="Theta 2",
    )
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    import numpyro.distributions as dist
    log_probs = dist.Gamma(
        final_params[..., 0].mean(axis=0),
        final_params[..., 1].mean(axis=0)
    ).log_prob(target)  # shape: (num_samples, num_data)
    nll = -jnp.mean(log_probs, axis=-1)
    print("Negative Log-Likelihood:", nll)
    
    posterior_param_samples_dict = model._get_posterior_param_samples()
    sns.set_style(style="white")
    for posterior_params_name, posterior_param_samples in (
            posterior_param_samples_dict.items()
    ):
        if not posterior_param_samples:
            continue  # Intercept may be None.

        if "weight" not in posterior_params_name:
            continue  # Only care about weights.

        feature_names = list({
            key.split("/")[0] for key in posterior_param_samples.keys()
        })
        for feature_name in feature_names:
            param_names = [
                key for key in posterior_param_samples.keys()
                if feature_name in key
            ]  # Parameters for this feature.
            for i, param_name in enumerate(param_names):
                layer_num = 4
                if f"dense_{layer_num}" not in param_name:
                    continue  # Let's only look at the last layer.
                if "scale" in param_name:
                    continue  # Not interested in the hierarchical prior.

                # Convert the posterior samples into a DataFrame for easier plotting.
                num_samples, num_layer_weights, num_outputs = (
                    posterior_param_samples[param_name].shape
                )
                samples_reshaped = posterior_param_samples[param_name].reshape(
                    num_samples,
                    num_layer_weights * num_outputs
                )
                column_names = [
                    f"Dense Layer {layer_num} - Neuron {i} (Output {j})"
                    for j in range(num_outputs)
                    for i in range(num_layer_weights)
                ]
                samples_df = pd.DataFrame(samples_reshaped, columns=column_names)

                # Create the plot of bivariate marginals for the last layer weights.
                for j in range(num_outputs):
                    for i in range(num_layer_weights - 1):
                        sns.jointplot(
                            x=f"Dense Layer {layer_num} - Neuron {i} (Output {j})",
                            y=f"Dense Layer {layer_num} - Neuron {i + 1} (Output {j})",
                            data=samples_df,
                            kind="kde",
                            alpha=0.5,
                            color="green"
                        ).plot_joint(sns.kdeplot, color="green", alpha=0.3)
                        plt.grid(True)
                plt.tight_layout()
                plt.show()

    pass