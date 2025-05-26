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

# --- DATA ---
def get_independent_synthetic_data(n_samples: int=1000, seed: int = 0):
    """
    Function to generate synthetic data for testing the BayesianNAM model.
    Returns a pandas DataFrame containing the synthetic data.

    Args:
        n_samples (int): Number of samples to generate.

    Returns:
        pd.DataFrame: DataFrame containing the synthetic data.
    """

    np.random.seed(seed)

    x1 = np.random.uniform(low=-1, high=1, size=n_samples)
    x2 = np.random.uniform(low=-1, high=1, size=n_samples)

    noise_parameters = {
        "x1": {"loc": 0, "scale": 0.01 + np.exp(x1)/10, "size": n_samples},
        "x2": {"loc": 0, "scale": 0.1 + np.square(x2), "size": n_samples},
    }

    noise = {
        "x1": np.random.normal(**noise_parameters["x1"]),
        "x2": np.random.normal(**noise_parameters["x2"]),
    }

    true_effects = {
        "x1": {
            "response": (x1)**2,
            "feature": x1,
            'noise_parameters': noise_parameters["x1"],
            'noise': noise["x1"]
        },
        "x2": {
            "response": np.sin(4*x2),
            "feature": x2,
            'noise_parameters': noise_parameters["x2"],
            'noise': noise["x2"]
        },
    }

    y = (
        true_effects["x1"]["response"] + true_effects["x1"]["noise"]
        + true_effects["x2"]["response"] + true_effects["x2"]["noise"]
    )

    return pd.concat(
        [
            pd.DataFrame(
                data={
                    'x1': x1,
                    'x2': x2,
                }
            ),
            pd.DataFrame(data={'Response': y})
        ], axis=1
    ), true_effects

def get_dependent_synthetic_data(n_samples: int=1000):
    """
    Function to generate synthetic data for testing the BayesianNAM model.
    Returns a pandas DataFrame containing the synthetic data.

    Args:
        n_samples (int): Number of samples to generate.

    Returns:
        pd.DataFrame: DataFrame containing the synthetic data.
    """

    np.random.seed(42)

    x1 = np.random.uniform(low=-1, high=1, size=n_samples)
    x2 = np.random.uniform(low=-1, high=1, size=n_samples)

    def v1(x):
        return  0.01 + np.exp(x1)/10
    def v2(x):
        return 0.1 + np.square(x2)

    noise_parameters = {
        "x": {"loc": 0, "scale": np.ones(n_samples), "size": n_samples},
    }

    a = np.sqrt(v1(x1))
    b = np.sqrt(v2(x2))
    alpha = a/(a + b)
    noise = {
        "x1": alpha * np.random.normal(
            **noise_parameters["x"]
        ) * a * b,
        "x2": (1 - alpha) * np.random.normal(
            **noise_parameters["x"]
        ) * a * b,
    }

    true_effects = {
        "x1": {
            "response": (x1)**2,
            "feature": x1,
            'noise_parameters': {"loc": 0, "scale": a**2*b/(a+b)}, # Std dev.
            'noise': noise["x1"]
        },
        "x2": {
            "response": np.sin(4*x2),
            "feature": x2,
            'noise_parameters': {"loc": 0, "scale": a*b**2/(a+b)}, # Std dev.
            'noise': noise["x2"]
        },
    }

    y = (
        true_effects["x1"]["response"] + true_effects["x1"]["noise"]
        + true_effects["x2"]["response"] + true_effects["x2"]["noise"]
    )

    return pd.concat(
        [
            pd.DataFrame(
                data={
                    'x1': x1,
                    'x2': x2,
                }
            ),
            pd.DataFrame(data={'Response': y})
        ], axis=1
    ), true_effects


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
        figsize=(2 * 6 * n_cols, 2 * 6 * len(single_true_effects))
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
            ax=ax_to_plot,
            alpha=0.5
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
            ax=ax_to_plot,
            alpha=0.5
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

    pass

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
            #         color=plt.cm.Reds(norm_counts[k]),
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
            figsize=(2* 6 * num_outputs, 2 * 6 * num_plots),
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
                        color=plt.cm.Reds(norm_counts[k]),
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

    # data, true_effects = get_independent_synthetic_data(n_samples=1000)
    # plot_synthetic_data(true_effects=true_effects)

    # X = data.drop(columns=['Response'])
    # y = data['Response']

    from sklearn.datasets import fetch_california_housing
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
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

    plot_feature_contributions(
        num_features=num_features,
        cat_features=cat_features,
        interaction_features=interaction_feature_information,
        submodel_contributions=submodel_contributions,
        num_outputs=num_outputs
    )

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
        log_probs, axis=-1
    )
    print("Negative Log-Likelihood:", nll)

    # assume group_by_chain is defined somewhere (e.g. True or False)
    posterior_param_samples_dict = model._get_posterior_param_samples(
        group_by_chain=True
    )

    # sns.set_style(style="white")
    # for layer_num in [0, 1, 2, 3, 4]:
    #     for posterior_params_name, posterior_param_samples in (
    #             posterior_param_samples_dict.items()
    #     ):
    #         if not posterior_param_samples:
    #             continue  # intercepts may be None
    #         if "weight" not in posterior_params_name:
    #             continue  # only care about weights
    #
    #         # feature names for this block of weights
    #         feature_names = list({
    #             key.split("/")[0]
    #             for key in posterior_param_samples.keys()
    #         })
    #         for feature_name in feature_names:
    #             # collect all param keys for this feature
    #             param_names = [
    #                 key for key in posterior_param_samples.keys()
    #                 if feature_name in key
    #             ]
    #
    #             for param_name in param_names:
    #                 # only plot the last dense layer’s kernel weights
    #                 if f"dense_{layer_num}" not in param_name:
    #                     continue
    #                 if "scale" in param_name:
    #                     continue
    #
    #                 raw = posterior_param_samples[param_name]
    #                 # detect chain‐grouped vs flattened by ndim
    #                 # flattened: (n_draws, n_weights, n_outputs)
    #                 # grouped:   (n_chains, n_draws, n_weights, n_outputs)
    #                 if raw.ndim == 4:
    #                     n_chains, n_draws, n_weights, n_outputs = raw.shape
    #                 else:
    #                     n_chains = 1
    #                     n_draws, n_weights, n_outputs = raw.shape
    #                     raw = raw[None, ...]  # add fake chain axis for uniformity
    #
    #                 # now raw has shape (n_chains, n_draws, n_weights, n_outputs)
    #                 for chain_idx in range(n_chains):
    #                     # reshape per‐chain
    #                     chain_samples = raw[chain_idx]
    #                     samples_reshaped = chain_samples.reshape(
    #                         n_draws, n_weights * n_outputs
    #                     )
    #                     column_names = [
    #                         f"Dense Layer {layer_num} - Neuron {i} (Out {j})"
    #                         for j in range(n_outputs)
    #                         for i in range(n_weights)
    #                     ]
    #                     samples_df = pd.DataFrame(
    #                         samples_reshaped, columns=column_names
    #                     )
    #
    #                     # title suffix
    #                     suffix = (
    #                         f" — Chain {chain_idx}"
    #                         if n_chains > 1 else ""
    #                     )
    #
    #                     if layer_num == 0:
    #                         from itertools import combinations
    #
    #                         for j1, j2 in combinations(range(n_outputs), 2):
    #                             x = samples_df.iloc[:, j1]
    #                             y = samples_df.iloc[:, j2]
    #
    #                             g = sns.JointGrid(x=x, y=y, height=6)
    #                             g.plot_joint(
    #                                 sns.kdeplot, fill=True,
    #                                 cmap="Greens", levels=10, alpha=0.8
    #                             )
    #                             g.plot_joint(
    #                                 sns.kdeplot, fill=False,
    #                                 color="green", linewidths=1.0, levels=10
    #                             )
    #                             g.plot_marginals(
    #                                 sns.kdeplot, fill=True, color="green"
    #                             )
    #
    #                             g.ax_joint.set_xlabel(
    #                                 f"Dense Layer {layer_num} - Kernel Weight {j1}"
    #                             )
    #                             g.ax_joint.set_ylabel(
    #                                 f"Dense Layer {layer_num} - Kernel Weight {j2}"
    #                             )
    #                             plt.suptitle(
    #                                 f"{feature_name} BNN Bivariate Marginal "
    #                                 f"Kernel Posterior{suffix}"
    #                             )
    #                             g.ax_joint.grid(True)
    #                             plt.tight_layout()
    #                             plt.show()
    #
    #                     else:
    #                         for out in range(n_outputs):
    #                             for k in range(n_weights):
    #                                 for l in range(k + 1, n_weights):
    #                                     x_vals = samples_df.iloc[:, k + out * n_weights]
    #                                     y_vals = samples_df.iloc[:, l + out * n_weights]
    #
    #                                     g = sns.JointGrid(
    #                                         data=samples_df,
    #                                         x=x_vals.name,
    #                                         y=y_vals.name,
    #                                         height=6
    #                                     )
    #                                     g.plot_joint(
    #                                         sns.kdeplot, fill=True,
    #                                         cmap="Greens", levels=10, alpha=0.8
    #                                     )
    #                                     g.plot_joint(
    #                                         sns.kdeplot, fill=False,
    #                                         color="green", linewidths=1.0, levels=10
    #                                     )
    #                                     g.plot_marginals(
    #                                         sns.kdeplot, fill=True, color="green"
    #                                     )
    #
    #                                     g.ax_joint.set_xlabel(
    #                                         f"Dense Layer {layer_num} - "
    #                                         f"Kernel Weight {k} (Out {out})"
    #                                     )
    #                                     g.ax_joint.set_ylabel(
    #                                         f"Dense Layer {layer_num} - "
    #                                         f"Kernel Weight {l} (Out {out})"
    #                                     )
    #                                     plt.suptitle(
    #                                         f"{feature_name} BNN Bivariate Marginal "
    #                                         f"Kernel Posterior{suffix}"
    #                                     )
    #                                     g.ax_joint.grid(True)
    #                                     plt.tight_layout()
    #                                     plt.show()
    #
    #         print(f"Layer {layer_num} done.")
    #     print("Marginal posteriors done.")

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from itertools import combinations

    posterior_param_samples_dict = model._get_posterior_param_samples(
        group_by_chain=False
    )

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

    #
    # # ------------------------------------------------------------------
    # # CONFIG FIR HADAMARD PRODUCT PLOTS
    # for layer_num in [1, 2, 3,]:
    #     for output_index in [0, 1]:
    #         # layer_num = 1  # last layer
    #         prev_layer_num = layer_num - 1  # one layer earlier
    #         # output_index = 0  # pick the first output head
    #         reduce_along_input = "mean"  # or "sum", "max", ...
    #         # ------------------------------------------------------------------
    #
    #         sns.set_style("white")
    #
    #         for posterior_params_name, posterior_param_samples in posterior_param_samples_dict.items():
    #             if not posterior_param_samples or "weight" not in posterior_params_name:
    #                 continue
    #
    #             feature_names = {key.split("/")[0] for key in posterior_param_samples}
    #             for feature_name in feature_names:
    #
    #                 # -------- pull the two kernels we need ---------------------------------
    #                 prev_key = f"{feature_name}/dense_{prev_layer_num}_kernel"
    #                 last_key = f"{feature_name}/dense_{layer_num}_kernel"
    #
    #                 if prev_key not in posterior_param_samples or last_key not in posterior_param_samples:
    #                     continue  # network might be smaller than we think
    #
    #                 W_prev = posterior_param_samples[prev_key]  # shape: (S, n_hidden, n_in)
    #                 W_last = posterior_param_samples[last_key]  # shape: (S, n_hidden, n_out)
    #
    #                 # -------- pick one output head & broadcast Hadamard --------------------
    #                 w_out = W_last[..., output_index]  # (S, n_hidden)
    #                 u = w_out[..., None] * W_prev  # (S, n_hidden, n_in)
    #
    #                 if reduce_along_input == "mean":
    #                     u = u.mean(axis=-1)  # (S, n_hidden)
    #                 elif reduce_along_input == "sum":
    #                     u = u.sum(axis=-1)
    #                 elif reduce_along_input == "max":
    #                     u = u.max(axis=-1)
    #                 else:
    #                     raise ValueError("Unknown reduction")
    #
    #                 # -------- DataFrame for seaborn ----------------------------------------
    #                 col_names = [f"u[{k}]" for k in range(u.shape[1])]
    #                 u_df = pd.DataFrame(u, columns=col_names)
    #
    #                 # -------- KDE plots (pairwise like Fig 2) ------------------------------
    #                 for k in range(u.shape[1]):
    #                     for l in range(k + 1, u.shape[1]):
    #                         g = sns.JointGrid(
    #                             data=u_df,
    #                             x=col_names[k],
    #                             y=col_names[l],
    #                             height=6
    #                         )
    #                         g.plot_joint(
    #                             sns.kdeplot,
    #                             cmap="Reds",
    #                             fill=True,
    #                             levels=10,
    #                             alpha=0.8
    #                         )
    #                         g.plot_joint(
    #                             sns.kdeplot,
    #                             color="red",
    #                             fill=False,
    #                             levels=10,
    #                             linewidths=1.0
    #                         )
    #                         g.plot_marginals(sns.kdeplot, color="red", fill=True)
    #
    #                         g.ax_joint.grid(True)
    #                         g.ax_joint.set_xlabel(f"Hadamard u[{k}]")
    #                         g.ax_joint.set_ylabel(f"Hadamard u[{l}]")
    #                         plt.suptitle(
    #                             f"{feature_name} – Hadamard Product "
    #                             f"(dense {prev_layer_num} ⊙ dense {layer_num})",
    #                             fontsize=12
    #                         )
    #                         plt.tight_layout()
    #                         plt.show()
    #         pass
    #
    #         # import cloudpickle
    #         # import lzma
    #         #
    #         # model_name = "small_synthetic_nam_100warmup_1000sample_8chains_16x16_relu_nonhierarchical_isotropic_0mean_5scale"
    #         # with lzma.open(
    #         #     filename=f"{model_name}.pkl.xz",
    #         #     mode="wb"
    #         # ) as f:
    #         #     cloudpickle.dump(model, f)
    #         #
    #         # with lzma.open(
    #         #     filename=f"{model_name}.pkl.xz",
    #         #     mode="rb"
    #         # ) as f:
    #         #     loaded_model = cloudpickle.load(f)
    #
    #     print(f"Layer {layer_num} done.")
    # print("Hadamard Marginal posteriors done.")

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

    pass
